import logictensornetworks as ltn
from scipy.spatial import distance
import tensorflow as tf
import numpy as np
import csv
import os
import math

ltn.default_layers = 5
ltn.default_smooth_factor = 1e-15
ltn.default_tnorm = "luk"
ltn.default_aggregator = "hmean"
ltn.default_positive_fact_penality = 0.
ltn.default_clauses_aggregator = "hmean"

data_training_dir = "data/train"
data_testing_dir = "data/test"
ontology_dir = "data/ontology"
ontology_dir = "data/ontology"

types = np.genfromtxt(os.path.join(ontology_dir, "classes.csv"), dtype="S", delimiter=",")
predicates = np.genfromtxt(os.path.join(ontology_dir, "predicates.csv"), dtype="S", delimiter=",")
selected_types = types[1:]
selected_predicates = predicates
number_of_features = len(types) + 4
number_of_extra_features = 7
objects = ltn.Domain(number_of_features, label="a_bounding_box")
pairs_of_objects = ltn.Domain(2*number_of_features + number_of_extra_features, label="a_pair_of_bounding_boxes")

import inspect
def is_of_type(obj_type, features):
    return tf.slice(features, [0, obj_type], [tf.shape(features)[0], 1])

isOfType = {}
isInRelation = {}
for t_idx, t in enumerate(selected_types):
    t_p = np.where(selected_types==t)[0][0] + 1
    isOfType[t] = ltn.Predicate("is_of_type_" + t.replace(" ", "_"), objects, layers=5, defined=lambda t_p, dom: is_of_type(t_p, dom), type_idx=t_p)

for p in selected_predicates:
    isInRelation[p] = ltn.Predicate(p.replace(" ", "_") + "_relation_", pairs_of_objects, layers=5)

objects_of_type = {}
objects_of_type_not = {}
object_pairs_in_relation = {}
object_pairs_not_in_relation = {}
for t in selected_types:
    objects_of_type[t] = ltn.Domain(number_of_features, label="objects_of_type_" + t.replace(" ", "_"))
    objects_of_type_not[t] = ltn.Domain(number_of_features, label="objects_of_type_not_" + t.replace(" ", "_"))
for p in selected_predicates:
    object_pairs_in_relation[p] = ltn.Domain(number_of_features*2 + number_of_extra_features,
                                             label="object_pairs_in_" + p.replace(" ", "_") + "_relation")
    object_pairs_not_in_relation[p] = ltn.Domain(number_of_features*2 + number_of_extra_features,
                                                 label="object_pairs_not_in_" + p.replace(" ", "_") + "_relation")

# arguments 2 vectors with xmin,ymin,xmax,ymax coordinates (2 bounding boxes at the image)
def computing_extended_features(bb1, bb2):

    # Area of bounding boxes
    rect_area1 = float((bb1[-2] - bb1[-4]) * (bb1[-1] - bb1[-3]))
    rect_area2 = float((bb2[-2] - bb2[-4]) * (bb2[-1] - bb2[-3]))

    # Area of intersected rectangle
    w_intersec = max(0, min([bb1[-2], bb2[-2]]) - max([bb1[-4], bb2[-4]]))
    h_intersec = max(0, min([bb1[-1], bb2[-1]]) - max([bb1[-3], bb2[-3]]))
    intersection_area = w_intersec * h_intersec

    # Centroids of rectangles CR1, CR2
    x_cr1 = (bb1[-2] + bb1[-4])/2.0
    y_cr1 = (bb1[-1] + bb1[-3])/2.0
    x_cr2 = (bb2[-2] + bb2[-4])/2.0
    y_cr2 = (bb2[-1] + bb2[-3])/2.0

    # Ratios with intersection area
    v1 = intersection_area/rect_area1
    v2 = intersection_area/rect_area2

    # Ratio of bounding boxes area
    v3 = rect_area1/rect_area2
    v4 = rect_area2/rect_area1

    v3_norm = (math.exp(rect_area1) - 1)/(math.exp(rect_area2 + 1) - 1)
    v4_norm = (math.exp(rect_area2) - 1)/(math.exp(rect_area1 + 1) - 1)

    # Euclidean distance
    v5_norm = distance.euclidean([x_cr1, y_cr1], [x_cr2, y_cr2])/math.sqrt(2)
    v5 = distance.euclidean([x_cr1, y_cr1], [x_cr2, y_cr2])

    # Angle between centroid1 and centroid2 antiClockWise
    angle = math.degrees(math.atan2(y_cr1 - y_cr2, x_cr2 - x_cr1))
    if angle >= 0:
        v6 = angle
    else:
        v6 = 360 + angle

    v7 = math.sin(math.radians(v6))
    v8 = math.cos(math.radians(v6))

    final_vec = [v1, v2, v3_norm, v4_norm, v5_norm, v7, v8]

    return final_vec


def normalize_data(data_dir, data):
    normalized_data = np.copy(data)
    width_height = np.genfromtxt(os.path.join(data_dir, "width_height.csv"), delimiter=",")
    normalized_data[:, -4] = normalized_data[:, -4]/width_height[:, 0]
    normalized_data[:, -3] = normalized_data[:, -3]/width_height[:, 1]
    normalized_data[:, -2] = normalized_data[:, -2]/width_height[:, 0]
    normalized_data[:, -1] = normalized_data[:, -1]/width_height[:, 1]
    return normalized_data


def get_data(train_or_test_switch, one_shot_features_flag, max_rows=10000000):
    #assert train_or_test_switch == "train" or train_or_test_switch == "test"

    # Fetching the data from the file system
    if train_or_test_switch == "train":
        data_dir = data_training_dir
    if train_or_test_switch == "test":
        data_dir = data_testing_dir
    if train_or_test_switch == "train_reduced_70":
        data_dir = "data/"+train_or_test_switch

    data = np.genfromtxt(os.path.join(data_dir, "features.csv"), delimiter=",", max_rows=max_rows)

    assert np.all(data[:, -4] < data[:, -2])
    assert np.all(data[:, -3] < data[:, -1])

    img_names = np.genfromtxt(os.path.join(data_dir, "features.csv"), delimiter=",", dtype=None, usecols=(0))
    idx_types_of_data = np.genfromtxt(os.path.join(data_dir, "types.csv"), dtype="i", max_rows=max_rows)
    types_of_data = types[idx_types_of_data]
    triples_s_o_p = np.genfromtxt(os.path.join(data_dir, "predicates.csv"), delimiter=",", dtype="i", max_rows=max_rows)

    if one_shot_features_flag:
        one_shot_features = np.zeros((data.shape[0], types.shape[0]))
        one_shot_features[np.arange(len(one_shot_features)), idx_types_of_data] = [1.0]
        data = np.hstack((data[:, 0, np.newaxis], one_shot_features, data[:, -4:]))

    data = normalize_data(data_dir, data)

    idx_of_cleaned_data = np.where(np.in1d(predicates[triples_s_o_p[:, -1]], selected_predicates))
    triples_s_o_p = triples_s_o_p[idx_of_cleaned_data]
    pairs_of_data = np.array([np.concatenate((data[s_o_p[0]][1:], data[s_o_p[1]][1:],
                                              computing_extended_features(data[s_o_p[0]], data[s_o_p[1]])))
                              for s_o_p in triples_s_o_p])

    set_sub_obj = set([tuple(sub_obj) for sub_obj in triples_s_o_p[:, :2]])
    unique_sub_obj = np.array([sub_obj for sub_obj in set_sub_obj])

    # Grouping bbs that belong to the same picture
    pics = {}
    pics_triples = {}
    for i in range(len(img_names)):
        triple_idxs = np.where(triples_s_o_p[:, 0] == i)[0]
        if img_names[i] in pics:
            pics[img_names[i]].append(i)
        else:
            pics[img_names[i]] = [i]

        if img_names[i] in pics_triples:
            pics_triples[img_names[i]] = np.vstack((pics_triples[img_names[i]], triples_s_o_p[triple_idxs]))
        else:
            pics_triples[img_names[i]] = triples_s_o_p[triple_idxs]

    cartesian_of_data = np.array(
        [np.concatenate((data[i][1:], data[j][1:], computing_extended_features(data[i], data[j]))) for p in
         pics for i in pics[p] for j in pics[p]])

    cartesian_of_bb_idxs = np.array([[i,j] for p in pics for i in pics[p] for j in pics[p]])

    print "End of loading data"

    return data, pairs_of_data, types_of_data, triples_s_o_p, cartesian_of_data, pics_triples, cartesian_of_bb_idxs


def get_vrd_ontology():
    is_subrelation_of = {}
    has_subrelations = {}
    inv_relations_of = {}
    not_relations_of = {}
    reflexivity = {}
    symmetry = {}
    range_relation = {}
    domain_relation = {}
    with open(os.path.join(ontology_dir, 'vrd_domain_ontology.csv')) as f:
        ontology_reader = csv.reader(f)
        for row in ontology_reader:
            domain_relation[row[0]] = row[1:]

    with open(os.path.join(ontology_dir, 'vrd_range_ontology.csv')) as f:
        ontology_reader = csv.reader(f)
        for row in ontology_reader:
            range_relation[row[0]] = row[1:]

    with open(os.path.join(ontology_dir, 'vrd_predicate_ontology.csv')) as f:
        ontology_reader = csv.reader(f)
        for row in ontology_reader:
            is_subrelation_of[row[0]] = []
            inv_relations_of[row[0]] = []
            not_relations_of[row[0]] = []

            for super_relation in row[1:]:
                if super_relation.split()[0] == 'inv':
                    not_relations_of[row[0]].append(super_relation[4:])
                    inv_relations_of[row[0]].append(super_relation[4:])
                elif super_relation.split()[0] == 'not':
                    not_relations_of[row[0]].append(super_relation[4:])
                elif super_relation.split()[0] == 'reflex':
                    reflexivity[row[0]] = True
                elif super_relation.split()[0] == 'irreflex':
                    reflexivity[row[0]] = False
                elif super_relation.split()[0] == 'symm':
                    symmetry[row[0]] = True
                elif super_relation.split()[0] == 'asymm':
                    symmetry[row[0]] = False
                else:
                    is_subrelation_of[row[0]].append(super_relation)
                    if super_relation in has_subrelations:
                        has_subrelations[super_relation].append(row[0])
                    else:
                        has_subrelations[super_relation] = [row[0]]

    return is_subrelation_of, has_subrelations, inv_relations_of, not_relations_of, reflexivity, symmetry, domain_relation, range_relation
