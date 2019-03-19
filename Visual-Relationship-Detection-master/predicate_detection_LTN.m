%   This file is for Predicting predicate 

%   Distribution code Version 1.0 -- Copyright 2016, AI lab @ Stanford University.
%   
%   The Code is created based on the method described in the following paper 
%   [1] "Visual Relationship Detection with Language Priors", 
%   Cewu Lu*, Ranjay Krishna*, Michael Bernstein, Li Fei-Fei, European Conference on Computer Vision, 
%   (ECCV 2016), 2016(oral). (* = indicates equal contribution)
%  
%   The code and the algorithm are for non-comercial use only.

%% data loading
addpath('evaluation');
load('data/objectListN.mat'); 
% given a object category index and ouput the name of it.

load('data/obj2vec.mat'); 
% word-to-vector embeding based on https://github.com/danielfrg/word2vec
% input a word and ouput a vector.

load('data/UnionCNNfeaPredicate.mat')
% the CNN score on union of the boundingboxes of the two participating objects in that relationship. 
% we provide our scores (VGG based) here, but you can re-train a new model.

load('data/objectDetRCNN.mat');
% object detection results. The scores are mapped into [0,1]. 
% we provide detected object (RCCN with VGG) here, but you can use a better model (e.g. ResNet).
% three items: 
% detection_labels{k}: object category index in k^{th} testing image.
% detection_bboxes{k}: detected object bounding boxes in k^{th} testing image. 
% detection_confs{k}: confident score vector in k^{th} testing image. 

load('data/Wb.mat');
% W and b in Eq. (2) in [1]

%% We assume we have ground truth object detection
% we will change "predicate" in rlp_labels_ours use our prediction

load('evaluation/gt.mat');
rlp_labels_ours = gt_tuple_label; 
sub_bboxes_ours = gt_sub_bboxes;
obj_bboxes_ours = gt_obj_bboxes;

model_label_list = {'KB_wc_10000', 'KB_nc_10000'};

for model_label_idx =1 : length(model_label_list)

    model_label = model_label_list{model_label_idx};
    fprintf('Computing results for model %s \n', model_label)
    load(['results_LTN/predicate_det_result_',model_label,'.mat']);

    %% computing Predicate Det. accuracy
    zeroShot100R = zeroShot_top_recall_Relationship(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
    zeroShot50R = zeroShot_top_recall_Relationship(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);

    fprintf('%0.2f\t%0.2f\n',100*zeroShot100R,100*zeroShot50R);
end