# LOGIC TENSOR NETWORKS FOR VISUAL RELATIONSHIP DETECTION

This repository contains the dataset, the source code and the models for the detection of visual relationships with [Logic Tensor Networks] (https://github.com/logictensornetworks/logictensornetworks).

## Introduction

Visual Relationship Detection is a task where the input is a simple image and the model is supposed to predict a set of relationships that are true about the image. There relationships are in the form of . Visual Relationship Detection is a natural extension to object detection where we now not only detect the objects in an image but predict how they are interacting with one another.

Detailed description of the task and our model is provided in our paper at ECCV 2016.

- This repository contains the implementation of the paper *Compensating Supervision Incompleteness with Prior Knowledge in Semantic Image Interpretation*, the generated grounded theories, Python and Matlab scripts for the evaluation and the Visual Relationship Dataset (VRD).
- You can use/test the trained grounded theories or train a new grounded theory, see how-tos below.

```
@InProceedings{donadello2019compensating,
   title = {Compensating Supervision Incompleteness with Prior Knowledge in Semantic Image Interpretation},
   author = {Donadello, Ivan and Serafini, Luciano},
   booktitle = {International Joint Conference on Neural Networks},
   year = to appear,
 }
```

## Using the Source Code

- The `models` folder will contain the multiclass and multilabel models after the training;
- The `history` folder will contain the accuracies and losses of the multiclass and multilabel models after the training;
- The `results` folder will contain the precision-recall curves results for the trained models after the evaluation.

- `code`: it contains the VRD and the associated logical constraints, the models folder, the evaluation code for VRD and the source code of LTNs.
    - `data`: this folder contains the VRD training and test set encoded for LTNs, the ontology that defines the logical constraints and the images in the test set. The VRD can be downloaded from https://cs.stanford.edu/people/ranjaykrishna/vrd/.
    - `models`: the trained grounded theories of the experiments;
    - `Visual-Relationship-Detection-master`: this is the evaluation code provided in https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection for the evaluation of the phrase, relationship and predicate detection tasks on the VRD.

**Requirements**

We train and test the models with the following software configuration. However, more recent versions of the libraries could also work:

- Ubuntu 14.04;
- Matlab R2014a;
- Python 2.7.6;
- TensorFlow 0.11;
- Numpy 1.13.1;
- Scikit-learn 0.18.1;
- Matplotlib 1.5.1;

**Training a grounded theory**

To run a train use the following command:
```sh
$ python train.py
```
- The trained grounded theories are saved in the `models` folder in the files `KB_nc_10000.ckpt` (no constraints) and `KB_wc_10000.ckpt` (with constraints). The number in the filename (`10000`) is a parameter in the code to set the number of iterations.

**Evaluating the grounded theories**

To run the evaluation use the following commands
```sh
$ python predicate_detection.py
$ python relationship_phrase_detection.py
```
Then, launch Matlab, move into the `Visual-Relationship-Detection-master` folder, execute the scripts `predicate_detection_LTN.m` and `relationship_phrase_detection_LTN.m` and see the results.





## The Food and Food Categories (FFoCat) Dataset

[Here](http://bit.do/eGcW5) you can download the `FFoCat.zip` file, unzip it in your local machine. The dataset is already divided into the `train` and `test` folder. The file `label.tsv` contains the food labels, the file `food_food_category_map.tsv` contains the food labels with the corresponding food category labels. 




