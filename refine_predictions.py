from anytree import Node, RenderTree, RenderTree, LevelOrderGroupIter
import rdflib
import os
from rdflib import URIRef
import numpy as np
from sets import Set

ontology_dir = "data/ontology"

def create_ontology_graph():
    # Construct ISA trees from triples

    graph = rdflib.Graph()
    graph.parse(os.path.join(ontology_dir, 'inferred_vrd'))

    ontology_labels_nodes = {}
    ontology_labels_equivalent_tmp = Set([])
    ontology_labels_equivalent = Set([])

    for s, p, o in graph.triples((None, URIRef("http://www.w3.org/2002/07/owl#equivalentProperty"), None)):
        # print s, " -> ", p, " -> ", o
        if "http://" in s and "http://" in o:

            subj_label = str(s.split("#")[1])
            obj_label = str(o.split("#")[1])
            ontology_labels_equivalent.add(subj_label)
            ontology_labels_equivalent.add(obj_label)

            if ontology_labels_nodes:
                new_node = True
                for node_label in ontology_labels_nodes.keys():

                    if subj_label in node_label.split(","):
                        ontology_labels_equivalent_tmp.remove(node_label)
                        ontology_labels_nodes[node_label].name = ontology_labels_nodes[node_label].name + "," + obj_label
                        ontology_labels_equivalent_tmp.add(ontology_labels_nodes[node_label].name)
                        ontology_labels_nodes[ontology_labels_nodes[node_label].name] = ontology_labels_nodes[
                            node_label]
                        del ontology_labels_nodes[node_label]
                        new_node = False

                    elif obj_label in node_label.split(","):
                        ontology_labels_equivalent_tmp.remove(node_label)
                        ontology_labels_nodes[node_label].name = ontology_labels_nodes[node_label].name + "," + subj_label
                        ontology_labels_equivalent_tmp.add(ontology_labels_nodes[node_label].name)
                        ontology_labels_nodes[ontology_labels_nodes[node_label].name] = ontology_labels_nodes[node_label]
                        del ontology_labels_nodes[node_label]
                        new_node = False
                if new_node:
                    ontology_labels_nodes[subj_label + "," + obj_label] = Node(subj_label + "," + obj_label)
                    ontology_labels_equivalent_tmp.add(subj_label + "," + obj_label)
            else:
                ontology_labels_nodes[subj_label + "," + obj_label] = Node(subj_label + "," + obj_label)
                ontology_labels_equivalent_tmp.add(subj_label + "," + obj_label)

    for s, p, o in graph.triples((None, URIRef("http://www.w3.org/2000/01/rdf-schema#subPropertyOf"), None)):
        #print s, " -> ", p, " -> ", o

        if "http://" in s and "http://" in o:

            subj_label = str(s.split("#")[1])
            obj_label = str(o.split("#")[1])

            subj_node_name = ""
            obj_node_name = ""
            for node_label in ontology_labels_equivalent_tmp:
                if subj_label in node_label.split(","):
                    subj_node_name = node_label
                    continue
                if obj_label in node_label.split(","):
                    obj_node_name = node_label
                    continue
            if subj_node_name and obj_node_name:
                ontology_labels_nodes[subj_node_name].parent = ontology_labels_nodes[obj_node_name]

            if subj_label not in ontology_labels_equivalent and obj_label not in ontology_labels_equivalent:

                if subj_label not in ontology_labels_nodes:
                    ontology_labels_nodes[subj_label] = Node(subj_label)
                if obj_label not in ontology_labels_nodes:
                    ontology_labels_nodes[obj_label] = Node(obj_label)

                ontology_labels_nodes[subj_label].parent = ontology_labels_nodes[obj_label]

            if subj_label in ontology_labels_equivalent and obj_label not in ontology_labels_equivalent:
                if obj_label not in ontology_labels_nodes:
                    ontology_labels_nodes[obj_label] = Node(obj_label)

                # retrieve subj node
                for node_label in ontology_labels_nodes.keys():
                    if subj_label in node_label.split(","):
                        ontology_labels_nodes[node_label].parent = ontology_labels_nodes[obj_label]

            if subj_label not in ontology_labels_equivalent and obj_label in ontology_labels_equivalent:
                if subj_label not in ontology_labels_nodes:
                    ontology_labels_nodes[subj_label] = Node(subj_label)

                # retrieve obj node
                for node_label in ontology_labels_nodes.keys():
                    if obj_label in node_label.split(","):
                        ontology_labels_nodes[subj_label].parent = ontology_labels_nodes[node_label]

    tree_list = []
    for node_label in ontology_labels_nodes:
        if ontology_labels_nodes[node_label].is_root:
            tree_list.append(ontology_labels_nodes[node_label])
    return tree_list, ontology_labels_equivalent_tmp


def aggregate_equiv(equiv_set, input_vec, predicate_dict, aggregator):
    for pair in equiv_set:
        aggregator_vec = []
        for pred in pair.split(","):
            aggregator_vec.append(input_vec[predicate_dict[pred]])

        if aggregator is "max":
            aggregator_value = np.max(aggregator_vec)
        elif aggregator is "min":
            aggregator_value = np.min(aggregator_vec)
        elif aggregator is "mean":
            aggregator_value = np.mean(aggregator_vec)

        for pred in pair.split(","):
            input_vec[predicate_dict[pred]] = aggregator_value

    return input_vec


def refine_equiv(values_of_predicates, selected_predicates, aggregator):
    _, equiv_set = create_ontology_graph()

    predicate_dict = {}
    for idx in range(len(selected_predicates)):
        predicate_dict[selected_predicates[idx].replace(" ", "_")] = idx

    for idx_pred in range(len(values_of_predicates)):
        values_of_predicates[idx_pred] = aggregate_equiv(equiv_set, values_of_predicates[idx_pred], predicate_dict, aggregator)

    return values_of_predicates