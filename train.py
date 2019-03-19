from visual_relationship_dataset import *
import tensorflow as tf
import logictensornetworks as ltn

ltn.default_optimizer = "rmsprop"

# swith between GPU and CPU
config = tf.ConfigProto(device_count={'GPU': 1})

number_of_positive_examples_types = 100
number_of_negative_examples_types = 100
number_of_positive_examples_predicates = 100
number_of_negative_examples_predicates = 100

# Load training data
train_data, pairs_of_train_data, types_of_train_data, triples_of_train_data, cartesian_of_train_data, _, cartesian_of_bb_idxs = get_data("train", True)

set_triples_of_train_data = set([(bb_pairs_idx[0], bb_pairs_idx[1]) for bb_pairs_idx in triples_of_train_data[:, :2]])
idxs_of_negative_examples = [idx for idx, pair in enumerate(cartesian_of_bb_idxs) if tuple(pair) not in set_triples_of_train_data]

# Computing positive and negative examples for predicates and types
idxs_of_positive_examples_of_predicates = {}
idxs_of_negative_examples_of_predicates = {}
idxs_of_positive_examples_of_types = {}

for type in selected_types:
    idxs_of_positive_examples_of_types[type] = np.where(types_of_train_data == type)[0]

for predicate in selected_predicates:
    idxs_of_positive_examples_of_predicates[predicate] = np.where(predicates[triples_of_train_data[:, -1]] == predicate)[0]
    idxs_of_negative_examples_of_predicates[predicate] = idxs_of_negative_examples

print "finished to upload and analyze data"
print "Start model definition"

# domain definition
clause_for_positive_examples_of_predicates = [
    ltn.Clause([ltn.Literal(True, isInRelation[p], object_pairs_in_relation[p])],
               label="examples_of_object_pairs_in_" + p.replace(" ", "_") + "_relation", weight=1.0) for p in selected_predicates]

clause_for_negative_examples_of_predicates = [
    ltn.Clause([ltn.Literal(False, isInRelation[p], object_pairs_not_in_relation[p])],
               label="examples_of_object_pairs_not_in_" + p.replace(" ", "_") + "_relation", weight=1.0) for p in selected_predicates]

# axioms from the Visual Relationship Ontology
isa_subrelation_of, has_subrelations, inv_relations_of, not_relations_of, reflexivity_relations, symmetry, domain_relation, range_relation = get_vrd_ontology()

so_domain = {}
os_domain = {}

for predicate in selected_predicates:
    so_domain[predicate] = ltn.Domain(number_of_features * 2 + number_of_extra_features, label="object_pairs_for_axioms")

for type in selected_types:
    so_domain[type] = ltn.Domain(number_of_features * 2 + number_of_extra_features, label="object_pairs_for_axioms")
    os_domain[type] = ltn.Domain(number_of_features * 2 + number_of_extra_features, label="inverse_object_pairs_for_axioms")

clauses_for_not_domain = [ltn.Clause([ltn.Literal(False, isInRelation[pred], so_domain[subj[4:]]),
                                      ltn.Literal(False, isOfType[subj[4:]], objects_of_type[subj[4:]])],
                                     label="not_domain_of_" + pred.replace(" ", "_"))
                          for pred in domain_relation.keys() for subj in domain_relation[pred]
                          if len(domain_relation[pred]) > 0 and subj.split(" ")[0] == "not"]

clauses_for_not_range = [ltn.Clause([ltn.Literal(False, isInRelation[pred], os_domain[obj[4:]]),
                                     ltn.Literal(False, isOfType[obj[4:]], objects_of_type[obj[4:]])],
                                    label="not_range_of_" + pred.replace(" ", "_"))
                         for pred in range_relation.keys() for obj in range_relation[pred]
                         if len(range_relation[pred]) > 0 and obj.split(" ")[0] == "not"]


def train(number_of_training_iterations=2500,
          frequency_of_feed_dict_generation=250,
          with_constraints=False,
          start_from_iter=1,
          saturation_limit=0.90):

    global idxs_of_positive_examples_of_predicates, idxs_of_negative_examples_of_predicates

    # defining the clauses of the background knowledge
    clauses = clause_for_positive_examples_of_predicates + clause_for_negative_examples_of_predicates

    if with_constraints:
        clauses = clauses + \
                  clauses_for_not_domain + \
                  clauses_for_not_range

# defining the label of the background knowledge
    for cl in clauses: print cl.label

    if with_constraints:
        kb_label = "KB_wc"
    else:
        kb_label = "KB_nc"

    # definition of the KB
    models_path = "models/"
    KB = ltn.KnowledgeBase(kb_label, clauses, models_path)

    # start training
    init = tf.initialize_all_variables()
    sess = tf.Session(config=config)
    if start_from_iter == 1:
        sess.run(init)
    if start_from_iter > 1:
        KB.restore(sess)

    feed_dict = get_feed_dict(idxs_of_positive_examples_of_predicates, idxs_of_negative_examples_of_predicates, idxs_of_positive_examples_of_types, with_constraints=with_constraints)
    train_kb = True

    for i in range(start_from_iter, number_of_training_iterations + 1):
        if i % frequency_of_feed_dict_generation == 0:
            if train_kb:
                print i
            else:
                train_kb = True
            if train_kb and (i==10000):
                KB.save(sess, version="_" + str(i))

            feed_dict = get_feed_dict(idxs_of_positive_examples_of_predicates,
                                      idxs_of_negative_examples_of_predicates,
                                      idxs_of_positive_examples_of_types,
                                      with_constraints=with_constraints)
            print "---- TRAIN",kb_label,"----"
        if train_kb:
            sat_level = sess.run(KB.tensor,feed_dict)

            if np.isnan(sat_level):
                train_kb = False
            if sat_level >= saturation_limit:
                train_kb = False
            else:
                KB.train(sess, feed_dict)
        print str(i) + ' --> ' + str(sat_level)

    print "end of training"
    sess.close()


def get_feed_dict(idxs_of_pos_ex_of_predicates, idxs_of_neg_ex_of_predicates, idxs_of_pos_ex_of_types, with_constraints=True):

    print "selecting new training data"
    feed_dict = {}

    # positive and negative examples for predicates
    for p in predicates:
        feed_dict[object_pairs_in_relation[p].tensor] = \
            pairs_of_train_data[np.random.choice(idxs_of_pos_ex_of_predicates[p], number_of_positive_examples_predicates)]

        feed_dict[object_pairs_not_in_relation[p].tensor] = \
            cartesian_of_train_data[np.random.choice(idxs_of_neg_ex_of_predicates[p], number_of_negative_examples_predicates)]

    # feed data for axioms
    if with_constraints:

        for predicate in predicates:
            feed_dict[so_domain[predicate].tensor] = feed_dict[object_pairs_in_relation[predicate].tensor]

        for t in selected_types:
            idxs_bb_type = np.random.choice(idxs_of_pos_ex_of_types[t], number_of_positive_examples_types)
            feed_dict[objects_of_type[t].tensor] = train_data[idxs_bb_type][:, 1:]

            idxs_bb_pairs_subj = []
            idxs_bb_pairs_obj = []

            for idx in idxs_bb_type:
                idxs_bb_pairs_subj.append(np.random.choice(np.where(cartesian_of_bb_idxs[:, 0] == idx)[0], 1)[0])
                idxs_bb_pairs_obj.append(np.random.choice(np.where(cartesian_of_bb_idxs[:, 1] == idx)[0], 1)[0])

            feed_dict[so_domain[t].tensor] = cartesian_of_train_data[idxs_bb_pairs_subj]
            feed_dict[os_domain[t].tensor] = cartesian_of_train_data[idxs_bb_pairs_obj]
    return feed_dict

if __name__ == "__main__":
    for wc in [True, False]:
        train(number_of_training_iterations=10000,
              frequency_of_feed_dict_generation=50,
              with_constraints=wc,
              start_from_iter=1,
              saturation_limit=.96)
