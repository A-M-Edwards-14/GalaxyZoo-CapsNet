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
Describe files
```abc```

## Acknowledgments
Project Acknowledgments
