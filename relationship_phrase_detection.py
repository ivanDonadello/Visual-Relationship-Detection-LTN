from visual_relationship_dataset import *
import os
import scipy.io as sio
from PIL import Image
import copy
from refine_predictions import refine_equiv

np.set_printoptions(precision=2)
np.set_printoptions(threshold=np.inf)

# swith between GPU and CPU
config = tf.ConfigProto(device_count={'GPU': 1})

img_dir = 'data/sg_test_images'

# Load training data for prio statistics on the dataset
_, _, _, triples_of_train_data, _, _, _ = get_data("train", True)

# Computing positive and negative examples for predicates and types
idxs_of_positive_examples_of_predicates = {}

for predicate in selected_predicates:
    idxs_of_positive_examples_of_predicates[predicate] = np.where(predicates[triples_of_train_data[:, -1]] == predicate)[0]

prior_stats = np.array([len(idxs_of_positive_examples_of_predicates[pred]) for pred in selected_predicates])
prior_freq = np.true_divide(prior_stats, np.sum(prior_stats))

image_path = sio.loadmat('Visual-Relationship-Detection-master/data/imagePath.mat')
object_detection = sio.loadmat('Visual-Relationship-Detection-master/data/objectDetRCNN.mat')
detection_bboxes = object_detection['detection_bboxes']
detection_labels = object_detection['detection_labels']
detection_confs = object_detection['detection_confs']

for img_id in range(len(detection_bboxes[0])):
    if len(detection_bboxes[0][img_id]) > 0:
        assert np.all(detection_bboxes[0][img_id][:, 0] < detection_bboxes[0][img_id][:, 2])
        assert np.all(detection_bboxes[0][img_id][:, 1] < detection_bboxes[0][img_id][:, 3])

features_detected_bb = []
obj_bboxes_ours = []
sub_bboxes_ours = []
obj_labels_ours = []
sub_labels_ours = []

semantic_feat_vect = np.zeros(len(types))

for pic_idx in range(detection_bboxes.shape[1]):
    detection_bboxes[0, pic_idx] = detection_bboxes[0, pic_idx].astype(np.float)
    features_per_image = np.empty((0, 2*number_of_features + number_of_extra_features))
    obj_bboxes_ours_per_image = np.array([]).reshape(0, 4)
    sub_bboxes_ours_per_image = np.array([]).reshape(0, 4)
    obj_label_per_image = np.array([])
    sub_label_per_image = np.array([])

    # normalize data
    if len(detection_bboxes[0, pic_idx]) > 0:
        img = Image.open(os.path.join(img_dir, image_path['imagePath'][0, pic_idx][0]).replace('png', 'jpg'))
        width, height = img.size
        normalized_detection_bboxes = copy.deepcopy(detection_bboxes)
        copy.deepcopy
        normalized_detection_bboxes[0, pic_idx][:, -4] /= width
        normalized_detection_bboxes[0, pic_idx][:, -3] /= height
        normalized_detection_bboxes[0, pic_idx][:, -2] /= width
        normalized_detection_bboxes[0, pic_idx][:, -1] /= height

    for bb1_idx in range(len(detection_bboxes[0, pic_idx])):
        for bb2_idx in range(len(detection_bboxes[0, pic_idx])):
            if bb1_idx != bb2_idx:
                bb1 = normalized_detection_bboxes[0, pic_idx][bb1_idx]
                bb2 = normalized_detection_bboxes[0, pic_idx][bb2_idx]
                sub_label_per_image = np.append(sub_label_per_image, detection_labels[0, pic_idx][bb1_idx, 0])
                obj_label_per_image = np.append(obj_label_per_image, detection_labels[0, pic_idx][bb2_idx, 0])

                feat_vect_bb1 = np.hstack((semantic_feat_vect, bb1))
                feat_vect_bb2 = np.hstack((semantic_feat_vect, bb2))
                feat_vect_bb1[detection_labels[0, pic_idx][bb1_idx]] = detection_confs[0, pic_idx][bb1_idx]
                feat_vect_bb2[detection_labels[0, pic_idx][bb2_idx]] = detection_confs[0, pic_idx][bb2_idx]
                feat_vec_pair = np.hstack((feat_vect_bb1, feat_vect_bb2, computing_extended_features(bb1, bb2)))

                features_per_image = np.vstack((features_per_image, feat_vec_pair[np.newaxis, :]))
                sub_bboxes_ours_per_image = np.vstack((sub_bboxes_ours_per_image, detection_bboxes[0, pic_idx][bb1_idx]))
                obj_bboxes_ours_per_image = np.vstack((obj_bboxes_ours_per_image, detection_bboxes[0, pic_idx][bb2_idx]))

    features_detected_bb.append(features_per_image)
    obj_bboxes_ours.append(obj_bboxes_ours_per_image)
    sub_bboxes_ours.append(sub_bboxes_ours_per_image)
    obj_labels_ours.append(obj_label_per_image)
    sub_labels_ours.append(sub_label_per_image)

model_list = [
    "models/KB_wc_2500.ckpt",
    "models/KB_nc_2500.ckpt"]

for model_type in model_list:

    model = model_type
    obj_bboxes_ours_output = []
    sub_bboxes_ours_output = []
    model_label = model.split("/")[-1][:-5]
    print model.upper()
    predicted_predicates_values_tensor = tf.concat(1, [isInRelation[predicate].tensor() for predicate in selected_predicates])
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    saver.restore(sess, model)
    rlp_confs_ours = []
    rlp_labels_ours = []

    for pic_idx in range(detection_bboxes.shape[1]):
        if pic_idx % 100 == 0:
            print "Eval img", pic_idx
        values_of_predicates = np.array([], dtype=np.float32).reshape(0, 70)
        values_of_predicates = sess.run(predicted_predicates_values_tensor, {pairs_of_objects.tensor: features_detected_bb[pic_idx]})

        values_of_predicates = refine_equiv(values_of_predicates, selected_predicates, "max")
        values_of_predicates = np.multiply(values_of_predicates, prior_freq)

        conf_predicates_per_image = values_of_predicates.flatten('F')
        sub_bboxes_ours_output.append(np.tile(sub_bboxes_ours[pic_idx], (len(selected_predicates), 1)))
        obj_bboxes_ours_output.append(np.tile(obj_bboxes_ours[pic_idx], (len(selected_predicates), 1)))
        label_predicates_per_image = np.hstack(
            (np.tile(sub_labels_ours[pic_idx], len(selected_predicates))[:, np.newaxis],
             np.repeat(np.array(range(1, len(selected_predicates) + 1)), len(features_detected_bb[pic_idx]))[:,
             np.newaxis],
             np.tile(obj_labels_ours[pic_idx], len(selected_predicates))[:, np.newaxis]))

        rlp_confs_ours.append(conf_predicates_per_image[:, np.newaxis])
        rlp_labels_ours.append(label_predicates_per_image)

    sess.close()

    sio.savemat("Visual-Relationship-Detection-master/results_LTN/relationship_det_result_" + model_label + ".mat",
                {'sub_bboxes_ours': sub_bboxes_ours_output,
                 'obj_bboxes_ours': obj_bboxes_ours_output,
                 'rlp_confs_ours': rlp_confs_ours,
                 'rlp_labels_ours': rlp_labels_ours})
