# LOGIC TENSOR NETWORKS FOR SEMANTIC IMAGE INTERPRETATION

- This repository contains the implementation of the paper *Compensating Supervision Incompleteness with Prior Knowledge in Semantic Image Interpretation*, the generated grounded theories, Python and Matlab scripts for the evaluation and the Visual Relationship Dataset (VRD).
- Before executing LTNs, install TensorFlow 0.11 library https://www.tensorflow.org/. The training and testing of the grounded theories have been performed on Ubuntu Linux 14.04 with Python 2.7.6 and Matlab R2014a.
- You can use/test the trained grounded theories or train a new grounded theory, see how-tos below.
- Download the repository, unzip the file `LTN_SII.zip` and move into the `LTN_SII/code` folder.

## Structure of the LTN_SII Folder

- `code`: it contains the VRD and the associated logical constraints, the models folder, the evaluation code for VRD and the source code of LTNs.
    - `data`: this folder contains the VRD training and test set encoded for LTNs, the ontology that defines the logical constraints and the images in the test set. The VRD can be downloaded from https://cs.stanford.edu/people/ranjaykrishna/vrd/.
    - `models`: the trained grounded theories of the experiments;
    - `Visual-Relationship-Detection-master`: this is the evaluation code provided in https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection for the evaluation of the phrase, relationship and predicate detection tasks on the VRD.

## How to Train a Grounded Theory

```sh
$ python train.py
```
- The trained grounded theories are saved in the `models` folder in the files `KB_nc_10000.ckpt` (no constraints) and `KB_wc_10000.ckpt` (with constraints).

## How to Evaluate the Grounded Theories

```sh
$ python predicate_detection.py
$ python relationship_phrase_detection.py
```
Then, launch Matlab, move into the `Visual-Relationship-Detection-master` folder, execute the scripts `predicate_detection_LTN.m` and `relationship_phrase_detection_LTN.m` and see the results.
