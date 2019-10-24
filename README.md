# LOGIC TENSOR NETWORKS FOR VISUAL RELATIONSHIP DETECTION

This repository contains the dataset, the source code and the models for the detection of visual relationships with [Logic Tensor Networks](https://github.com/logictensornetworks/logictensornetworks).

## Introduction

Semantic Image Interpretation is the task of extracting a structured semantic description from images. This requires the detection of *visual relationships*: triples (subject, relation, object) describing a semantic relation between the bounding box of a subject and the bounding box of an object. Here, we perform the detection of visual relationships by using Logic Tensor Networks (LTNs), a novel Statistical Relational Learning framework that exploits both the similarities with other seen relationships and background knowledge, expressed with logical constraints between subjects, relations and objects. The experiments are conducted on the Visual Relationship Dataset (VRD).

A detailed description of the work is provided in our paper *Compensating Supervision Incompleteness with Prior Knowledge in Semantic Image Interpretation* at IJCNN 2019:
```
 @inproceedings{donadello2019compensating,
  author    = {Ivan Donadello and
               Luciano Serafini},
  title     = {Compensating Supervision Incompleteness with Prior Knowledge in Semantic
               Image Interpretation},
  booktitle = {{IJCNN}},
  pages     = {1--8},
  publisher = {{IEEE}},
  year      = {2019}
}
```
[Here](https://www.youtube.com/watch?v=y2-altg3FFw) a video shows a demo of the system.

## Using the Source Code

- The `data` folder contains the LTNs encoding of the VRD training and test set, the ontology that defines the logical constraints and the images of the VRD test set. Images and their annotations can be downloaded from https://cs.stanford.edu/people/ranjaykrishna/vrd/.
- The `models` folder contains the trained grounded theories of the experiments;
- The `Visual-Relationship-Detection-master` folder contains the object detector model and the evaluation code provided in https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection for the evaluation of the phrase, relationship and predicate detection tasks on the VRD.

**Requirements**

We train and test the grounded theories with the following software configuration. However, more recent versions of the libraries could also work:

- Ubuntu 14.04;
- Matlab R2014a;
- Python 2.7.6;
- TensorFlow 0.11.0;
- Numpy 1.13.1;
- Scikit-learn 0.18.1;
- Matplotlib 1.5.1;

**Training a grounded theory**

To run a train use the following command:
```sh
$ python train.py
```
- The trained grounded theories are saved in the `models` folder in the files `KB_nc_2500.ckpt` (no constraints) and `KB_wc_2500.ckpt` (with constraints). The number in the filename (`2500`) is a parameter in the code to set the number of iterations.

**Evaluating the grounded theories**

To run the evaluation use the following commands
```sh
$ python predicate_detection.py
$ python relationship_phrase_detection.py
```
Then, launch Matlab, move into the `Visual-Relationship-Detection-master` folder, execute the scripts `predicate_detection_LTN.m` and `relationship_phrase_detection_LTN.m` and see the results.
