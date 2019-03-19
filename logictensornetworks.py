import sys
import tensorflow as tf
import numpy as np
import pdb

default_layers = 10
default_smooth_factor = 0.0000001
default_tnorm = "product"
default_optimizer = "gd"
default_aggregator = "min"
default_positive_fact_penality = 1e-6
default_clauses_aggregator = "min"


def train_op(loss, optimization_algorithm):
    if optimization_algorithm == "ftrl":
        optimizer = tf.train.FtrlOptimizer(learning_rate=0.01, learning_rate_power=-0.5)
    if optimization_algorithm == "gd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
    if optimization_algorithm == "ada":
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    if optimization_algorithm == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9)
    return optimizer.minimize(loss)


def PR(tensor):
    global count
    np.set_printoptions(threshold=np.nan)
    return tf.Print(tensor, [tf.shape(tensor), tensor.name,tensor], summarize=200000)


def disjunction_of_literals(literals, label="no_label"):
    list_of_literal_tensors = [lit.tensor for lit in literals]
    literals_tensor = tf.concat(1,list_of_literal_tensors)
    if default_tnorm == "product":
        result = 1.0-tf.reduce_prod(1.0-literals_tensor, 1, keep_dims=True)
    if default_tnorm == "yager2":
        result = tf.minimum(1.0, tf.sqrt(tf.reduce_sum(tf.square(literals_tensor), 1, keep_dims=True)))
    if default_tnorm == "luk":
        print "data aggregator is lukas"
        result = tf.minimum(1.0, tf.reduce_sum(literals_tensor, 1, keep_dims=True))
        PR(result)
    if default_tnorm == "goedel":
        result = tf.reduce_max(literals_tensor, 1, keep_dims=True, name=label)
    if default_aggregator == "product":
        return tf.reduce_prod(result, keep_dims=True)
    if default_aggregator == "mean":
        print "data aggregator is mean"
        return tf.reduce_mean(result, keep_dims=True, name=label)
    if default_aggregator == "gmean":
        return tf.exp(tf.mul(tf.reduce_sum(tf.log(result), keep_dims=True),
                             tf.inv(tf.to_float(tf.size(result)))), name=label)
    if default_aggregator == "hmean":
        print "data aggregator is hmean"
        return tf.div(tf.to_float(tf.size(result)), tf.reduce_sum(tf.inv(result), keep_dims=True))
    if default_aggregator == "min":
        print "data aggregator is min"
        return tf.reduce_min(result, keep_dims=True, name=label)
    if default_aggregator == "qmean":
        print "data aggregator is qmean"
        return tf.sqrt(tf.reduce_mean(tf.square(result), keep_dims=True), name=label)
    if default_aggregator == "cmean":
        print "data aggregator is cmean"
        return tf.pow(tf.reduce_mean(tf.pow(result, 3), keep_dims=True), tf.inv(tf.to_float(3)), name=label)


def smooth(parameters):
    norm_of_omega = tf.reduce_sum(tf.expand_dims(tf.concat(0, [tf.expand_dims(tf.reduce_sum(tf.square(par)), 0)
                                                               for par in parameters]), 1))
    return tf.mul(default_smooth_factor, norm_of_omega)


class Domain:
    def __init__(self,columns, dom_type="float", label=None):
        self.columns = columns
        self.label = label
        self.tensor = tf.placeholder(dom_type, shape=[None, self.columns], name=self.label)
        self.parameters = []


class Domain_concat(Domain):

    def __init__(self, domains):
        self.columns = np.sum([dom.columns for dom in domains])
        self.label = "concatenation of" + ",".join([dom.label for dom in domains])
        self.tensor = tf.concat(1, [dom.tensor for dom in domains])
        self.parameters = [par for dom in domains for par in dom.parameters]


class Domain_slice(Domain):

    def __init__(self, domain, begin_column, end_column):
        self.columns = end_column - begin_column
        self.label = "projection of" + domain.label + "from column "+begin_column + " to column " + end_column
        self.tensor = tf.concat(1, tf.split(1, domain.columns, domain.tensor)[begin_column:end_column])
        self.parameters = domain.parameters


class Function(Domain):
    def __init__(self, label, domain, range, value=None):
        self.label = label
        self.domain = domain
        self.range = range
        self.value = value
        if self.value:
            self.parameters = []
        else:
            self.M = tf.Variable(tf.random_normal([self.domain.columns,
                                                   self.range.columns]),
                                 name="M_"+self.label)

            self.n = tf.Variable(tf.random_normal([1, self.range.columns]),
                                 name="n_"+self.label)
            self.parameters = [self.n, self.M]
        if self.value:
            self.tensor = self.value
        else:
            self.tensor = tf.add(tf.matmul(self.domain, self.M), self.n)


class Predicate:
    def __init__(self, label, domain, layers=default_layers, defined=None, type_idx=None):
        self.label = label
        self.type_idx = type_idx
        self.defined = defined
        self.domain = domain
        self.number_of_layers = layers
        self.W = tf.Variable(tf.zeros([layers,
                                              self.domain.columns,
                                              self.domain.columns]),
                             name="W"+label)
        self.V = tf.Variable(tf.zeros([layers,
                                               self.domain.columns]),
                             name="V"+label)
        self.b = tf.Variable(tf.neg(tf.ones([1,layers])),
                             name="b"+label)
        self.u = tf.Variable(tf.ones([layers,1]),
                             name="u"+label)
        self.parameters = [self.W, self.V, self.b, self.u]

    def tensor(self, domain=None):
        if self.defined is not None:
            if domain is None:
                return self.defined(self.type_idx, self.domain.tensor)
            else:
                return self.defined(self.type_idx, domain.tensor)
        if domain is None:
            domain = self.domain
        X = domain.tensor
        XW = tf.batch_matmul(tf.tile(tf.expand_dims(X, 0), [self.number_of_layers, 1, 1]), self.W)
        XWX = tf.squeeze(tf.batch_matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])))
        XV = tf.matmul(X, tf.transpose(self.V))
        gX = tf.matmul(tf.tanh(XWX + XV + self.b), self.u)
        return tf.sigmoid(gX)


class Literal:
    def __init__(self, polarity, predicate, domain=None):
        self.predicate = predicate
        self.polarity = polarity
        if domain is None:
            self.domain = predicate.domain
        else:
            self.domain = domain
        if polarity:
            self.tensor = predicate.tensor(domain)
        else:
            if default_tnorm == "product" or default_tnorm == "goedel":
                y = tf.equal(predicate.tensor(domain), 0.0)
                self.tensor = tf.cast(y, tf.float32)
            if default_tnorm == "yager2":
                self.tensor = 1-predicate.tensor(domain)
            if default_tnorm == "luk":
                self.tensor = 1-predicate.tensor(domain)

        self.parameters = predicate.parameters + domain.parameters


class Clause:
    def __init__(self, literals, label=None, weight=1.0):
        self.weight = weight
        self.label = label
        self.literals = literals
        self.tensor = disjunction_of_literals(self.literals, label=label)
        self.predicates = set([lit.predicate for lit in self.literals])
        self.parameters = [par for lit in literals for par in lit.parameters]


class KnowledgeBase:

    def __init__(self, label, clauses, save_path=""):
        print "defining the knowledge base", label
        self.label = label
        self.clauses = clauses
        self.parameters = [par for cl in self.clauses for par in cl.parameters]
        if not self.clauses:
            self.tensor = tf.constant(1.0)
        else:
            clauses_value_tensor = tf.concat(0, [cl.tensor for cl in clauses])
            if default_clauses_aggregator == "min":
                print "clauses aggregator is min"
                self.tensor = tf.reduce_min(clauses_value_tensor)
            if default_clauses_aggregator == "mean":
                print "clauses aggregator is mean"
                self.tensor = tf.reduce_mean(clauses_value_tensor)
            if default_clauses_aggregator == "hmean":
                print "clauses aggregator is hmean"
                self.tensor = tf.div(tf.to_float(tf.size(clauses_value_tensor)), tf.reduce_sum(tf.inv(clauses_value_tensor), keep_dims=True))
            if default_clauses_aggregator == "wmean":
                print "clauses aggregator is weighted mean"
                weights_tensor = tf.constant([cl.weight for cl in clauses])
                self.tensor = tf.div(tf.reduce_sum(tf.mul(weights_tensor, clauses_value_tensor)), tf.reduce_sum(weights_tensor))
        if default_positive_fact_penality != 0:
            self.loss = smooth(self.parameters) + \
                        tf.mul(default_positive_fact_penality, self.penalize_positive_facts()) - \
                        PR(self.tensor)
        else:
            self.loss = smooth(self.parameters) - PR(self.tensor)
        self.save_path = save_path
        self.train_op = train_op(self.loss, default_optimizer)
        self.saver = tf.train.Saver(max_to_keep=20)
        print "knowledge base", label, "is defined"

    def penalize_positive_facts(self):
        tensor_for_positive_facts = [tf.reduce_sum(Literal(True, lit.predicate, lit.domain).tensor, keep_dims=True) for cl in self.clauses for lit in cl.literals]
        return tf.reduce_sum(tf.concat(0, tensor_for_positive_facts))

    def save(self, sess, version=""):
        save_path = self.saver.save(sess, self.save_path + self.label+version+".ckpt")

    def restore(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("restoring model")
            self.saver.restore(sess, ckpt.model_checkpoint_path)

    def train(self, sess, feed_dict={}):
        return sess.run(self.train_op, feed_dict)

    def is_nan(self, sess, feed_dict={}):
        return sess.run(tf.is_nan(self.tensor), feed_dict)
