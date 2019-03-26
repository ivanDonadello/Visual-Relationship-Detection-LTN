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
gt = sio.loadmat('Visual-Relationship-Detection-master/evaluation/gt.mat')
gt_sub_bboxes = gt['gt_sub_bboxes']
gt_obj_bboxes = gt['gt_obj_bboxes']
gt_tuple_label = gt['gt_tuple_label']

for img_id in range(len(gt_sub_bboxes[0])):
    if len(gt_sub_bboxes[0][img_id]) > 0:
        assert np.all(gt_sub_bboxes[0][img_id][:, 0] < gt_sub_bboxes[0][img_id][:, 2])
        assert np.all(gt_sub_bboxes[0][img_id][:, 1] < gt_sub_bboxes[0][img_id][:, 3])

        assert np.all(gt_obj_bboxes[0][img_id][:, 0] < gt_obj_bboxes[0][img_id][:, 2])
        assert np.all(gt_obj_bboxes[0][img_id][:, 1] < gt_obj_bboxes[0][img_id][:, 3])

features_detected_bb = []
obj_bboxes_ours = []
sub_bboxes_ours = []
obj_labels_ours = []
sub_labels_ours = []

semantic_feat_vect = np.zeros(len(types))

for pic_idx in range(gt_tuple_label.shape[1]):
    gt_sub_bboxes[0, pic_idx] = gt_sub_bboxes[0, pic_idx].astype(np.float)
    gt_obj_bboxes[0, pic_idx] = gt_obj_bboxes[0, pic_idx].astype(np.float)
    features_per_image = np.empty((0, 2*number_of_features + number_of_extra_features))
    obj_bboxes_ours_per_image = np.array([]).reshape(0, 4)
    sub_bboxes_ours_per_image = np.array([]).reshape(0, 4)
    obj_label_per_image = np.array([])
    sub_label_per_image = np.array([])

    # normalize data
    if len(gt_sub_bboxes[0, pic_idx]) > 0:
        img = Image.open(os.path.join(img_dir, image_path['imagePath'][0, pic_idx][0]).replace('png', 'jpg'))
        width, height = img.size
        normalized_gt_sub_bboxes = copy.deepcopy(gt_sub_bboxes[0, pic_idx])
        normalized_gt_sub_bboxes[:, -4] /= width
        normalized_gt_sub_bboxes[:, -3] /= height
        normalized_gt_sub_bboxes[:, -2] /= width
        normalized_gt_sub_bboxes[:, -1] /= height

        normalized_gt_obj_bboxes = copy.deepcopy(gt_obj_bboxes[0, pic_idx])
        normalized_gt_obj_bboxes[:, -4] /= width
        normalized_gt_obj_bboxes[:, -3] /= height
        normalized_gt_obj_bboxes[:, -2] /= width
        normalized_gt_obj_bboxes[:, -1] /= height

    for bb_idx in range(len(gt_tuple_label[0, pic_idx])):
        bb1 = normalized_gt_sub_bboxes[bb_idx]
        bb2 = normalized_gt_obj_bboxes[bb_idx]
        sub_label_per_image = np.append(sub_label_per_image, gt_tuple_label[0, pic_idx][bb_idx, 0])
        obj_label_per_image = np.append(obj_label_per_image, gt_tuple_label[0, pic_idx][bb_idx, 2])

        feat_vect_bb1 = np.hstack((semantic_feat_vect, bb1))
        feat_vect_bb2 = np.hstack((semantic_feat_vect, bb2))
        feat_vect_bb1[gt_tuple_label[0, pic_idx][bb_idx, 0]] = 1.0
        feat_vect_bb2[gt_tuple_label[0, pic_idx][bb_idx, 2]] = 1.0
        feat_vec_pair = np.hstack((feat_vect_bb1, feat_vect_bb2, computing_extended_features(bb1, bb2)))

        features_per_image = np.vstack((features_per_image, feat_vec_pair[np.newaxis, :]))
        sub_bboxes_ours_per_image = np.vstack((sub_bboxes_ours_per_image, gt_sub_bboxes[0, pic_idx][bb_idx]))
        obj_bboxes_ours_per_image = np.vstack((obj_bboxes_ours_per_image, gt_obj_bboxes[0, pic_idx][bb_idx]))

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
    model_label = model.split("/")[-1][:-5]
    print model.upper()
    obj_bboxes_ours_output = []
    sub_bboxes_ours_output = []
    predicted_predicates_values_tensor = tf.concat(1, [isInRelation[predicate].tensor() for predicate in selected_predicates])
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    saver.restore(sess, model)
    rlp_confs_ours = []
    rlp_labels_ours = []
    error_an = np.array([]).reshape(0, 8)

    for pic_idx in range(gt_tuple_label.shape[1]):
        if pic_idx % 100 == 0:
            print "Eval img", pic_idx
        values_of_predicates = sess.run(predicted_predicates_values_tensor, {pairs_of_objects.tensor: features_detected_bb[pic_idx]})

        values_of_predicates = refine_equiv(values_of_predicates, selected_predicates, "max")
        values_of_predicates = np.multiply(values_of_predicates, prior_freq)

        conf_predicates_per_image = values_of_predicates.flatten('F')
        sub_bboxes_ours_output.append(np.tile(sub_bboxes_ours[pic_idx], (len(selected_predicates), 1)))
        obj_bboxes_ours_output.append(np.tile(obj_bboxes_ours[pic_idx], (len(selected_predicates), 1)))
        # Matlab indices start from 1
        label_predicates_per_image = np.hstack((np.tile(sub_labels_ours[pic_idx], len(selected_predicates))[:, np.newaxis],
                                                np.repeat(np.array(range(1, len(selected_predicates) + 1)), len(features_detected_bb[pic_idx]))[:, np.newaxis],
                                                np.tile(obj_labels_ours[pic_idx], len(selected_predicates))[:, np.newaxis]))

        rlp_confs_ours.append(conf_predicates_per_image[:, np.newaxis])
        rlp_labels_ours.append(label_predicates_per_image)
    sess.close()

    sio.savemat("Visual-Relationship-Detection-master/results_LTN/predicate_det_result_" + model_label + ".mat",
                {'sub_bboxes_ours': sub_bboxes_ours_output,
                 'obj_bboxes_ours': obj_bboxes_ours_output,
                 'rlp_confs_ours': rlp_confs_ours,
                 'rlp_labels_ours': rlp_labels_ours})
