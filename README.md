# GalaxyZoo_CapsNet

## Table of contents
* [Project Overview](#general-info)
* [Required Packages and Technologies](#technologies)
* [Data](#setup)
* [Files](#files)
* [Acknowledgments](#acknowledgments)

## Project Overview
This project investigated the relative abilities of a capsule network and residual network to reproduce the human classifications of galaxy images. This repository provides code that can process the required image data into the necessary data frame, train both models, evaluate their classification accuracies and test their abilities to reproduce well known physical results related to galaxy evolution.
	
## Required Packages and Technologies
The required packages for this project are:
* Pytorch
* CUDA
* Torchvision
* Sys
* SKimage
* SKlearn
* PIL
* Scipy
* Matplotlib
* Seaborn
* Numpy
* Pandas

This project also used 4 CPUs and 1 Tesla V100 GPU on the Lancaster University [High End Computing (HEC) Cluster](https://answers.lancaster.ac.uk/display/ISS/High+End+Computing+%28HEC%29+help).
	
## Data
All datasets used in this project can be downloaded from the following locations:

* The SDSS galaxy images and their corresponding labels used can be found on [Kaggle](https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/data) which provided 61578 RGB images in total. For access to the complete labelled dataset of SDSS galaxy images refer to the [Galaxy Zoo](https://data.galaxyzoo.org/).

* The higher resolution DECaLS galaxy images and their corresponding labels can be found on [Zenodo](https://zenodo.org/record/4196267#.YqiMJqHMLIU).

* The sample dataset of galaxy colours and total stellar mass was taken from [Schawinski et al. 2010b](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJ/711/284#/browse). For the complete dataset of galaxy colours refer to [MPA-JHU](https://www.sdss.org/dr12/spectro/galaxy_mpajhu/) and [NYU VAGC](http://sdss.physics.nyu.edu/vagc/) for the complete dataset of galaxy stellar masses.

* The galaxy Sersic indicies were downloaded from [Simard et al. 2011](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/196/11#/browse).



## Files
### CapsNet
The CapsNet folder contains 3 versions of the capsule network code: ```CapsNetReconstructor.py```, ```CapsNetRegressor.py``` and ```CapsNetPredictor.py```

```CapsNetRegressor.py``` is used to train the capsule network to predict the Galaxy Zoo vote fractions corresponding to an image. It accepts input data in the form of a tensor [Number of images, Number of colour channels, Image width, Image height] and matches each image in the tensor, by index, to the image label tensor [Number of images, Number of vote fractions]. The network uses an adam optimizer to minimize the mean squared error between the actual vote fractions and the network predicted fractions. The network will output the average value of the mean squared error across all image at each epoch, as well as saving the trained set of weights to the epoch_%d.pt file.

The ```CapsNetPredictor.py``` allows you to load in the pre-trained weights from the epoch_%d.pt file to predict the vote fractions corresponding to a set of input images to the network. 

```CapsNetReconstructor.py``` trains a capsule network to classify a galaxy image as either smooth and rounded, featured or an artefact using binary labels rather than vote fractions. Using the trained set of weights, the network reconstructs the images giving a visualisation of the features that the capsule network is able to detect to classify images. 


Note:

* If training or classifying RGB images change ‘in_channels = 3’, if using greyscale images set ‘in_channels =1’.

* ```CapsNetPredictor.py``` code failed to work when classifying 1 image with it outputting a 16-dimensional vector rather than the predicted array of vote fractions. However, it works fine when classifying more than one image.


### DataAnalysis
AccuracyPlot.py

ColourBar_Plot.py

Colour_Mass_Plot.py

HistogramPlot.py

KS_Test.py

ROC_BinaryLabel.py

ROC_Plotter.py

ReconstructImages.py

SersicVotes_Errors.py

### Dataloader
The code within the Dataloader folder is used to convert a folder of images into a suitable tensor that can be fed as an input of image data to either the CapsNet or ResNet. 

For both the ```Segmenter_Dataloader.py``` and ```Dataloader.py``` a directory that will contain all the galaxy images must be specified (‘root_dir=’), as well as the relative file paths/names of each image in that directory. The CSV file, loaded as ‘csv_file=’, must have the first column containing the relative file paths/names of all the images in the image directory. It is also useful to have the other columns in this CSV file corresponding to the vote fractions for each galaxy image. A series of transforms are applied to each image in order to crop them and convert them into a tensor format. Each image tensor is then appended to a list, such that each entry in the list corresponds to a single image tensor. The end result is an .npy file containing all the image data with a shape: [Number of images, Number of colour channels, Image width, Image Height].

The ```Segmenter_Dataloader.py``` works in an identical manner, expect the transforms applied to each image also includes Otsu’s thresholding method which acts to remove the image background.



### Miscellaneous


### ResNet
The ResNet folder contains 4 files: ```ResNetRGB.py```, ```ResNetRGBPredict.py```, ```ResNetGrey.py``` and ```ResNetGreyPredict.py```.

```ResNetRGB.py``` and ```ResNetGrey.py``` are the two sets of code used to train the ResNet model in a similar manner to the ```CapsNetRegressor.py```. Both ResNet models include data augmentation such as a series of horizontal and vertical flips, as well as 45-degree rotations, which effectively quadruples the size of the dataset. The trained weights are saved into the ‘epoch_.pt’ file at the end, these pre-trained weights can then be loaded into either ```ResNetRGBPredict.py``` or ```ResNetGreyPredict.py``` to obtain the predicted vote fractions corresponding to a galaxy image (whether that image is an RGB or greyscale image).
 

## Acknowledgments
Project Acknowledgments
