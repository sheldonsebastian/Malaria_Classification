# Red Blood Cell Classification

Sheldon Sebastian

![](saved_images/banner.jpg)
<center>Photo by <a href="https://unsplash.com/@switch_dtp_fotografie?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Lucas van Oort</a> on <a href="https://unsplash.com/s/photos/mosquito?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a></center>

## Abstract

<div style="text-align: justify"> 

The goal of this project is to identify whether a red blood cell is healthy or infected with any of the following stages of malaria: ring, schizont and trophozoite using a Multi-Layer Perceptron (MLP). Custom MLP was created and hyper-parameter tuning was performed on it using manual search, random grid search and automatic hyper-parameter tuning using Optuna.  
</div>
<br>
<i>Keywords</i>: Computer Vision, Deep Learning, Image Classification, Multi Layer Perceptron

## Table of Contents:

- Introduction
- Dataset Description and Data Preprocessing	
- Exploratory Data Analysis
- Modeling and Evaluation Metric
- Results & Analysis	
- Conclusion
- Future Work	
- References

## Introduction

<div style="text-align: justify">
Malaria is a mosquito-borne disease caused by a parasite. About 2,000 cases of malaria are diagnosed in the United States each year.<sup>[1]</sup> People with malaria often experience fever, chills, and flu-like illness. Left untreated, they may develop severe complications and die. For this project a Multi-Layer Perceptron (MLP) was used to classify blood cells into 4 different categories: red blood cell, ring, schizont and trophozoite. An image labelled as red blood cell is healthy and the rest of the labels indicate some stage of malaria infection. A custom MLP architecture was constructed and hyper-parameter tuning was performed manually, using random-grid search and automatically using Optuna<sup>[2]</sup>. <br><br>If a model can successfully identify these types of cells from images taken with a microscope, this would allow to automate a very time-consuming testing process, leaving human doctors with more time to treat the actual disease. Furthermore, an early malaria detection can save lives!   
</div>

## Dataset Description and Data Preprocessing
<div style="text-align: justify">
The dataset was provided by <a href="https://www.linkedin.com/in/amir-jafari-phd-5a153863/">Dr. Amir Jafari</a> and is available <a href = "https://github.com/sheldonsebastian/Red-Blood-Cell-Classification/tree/main/input/train">here</a>. The dataset contains 8,607 unique png files, and their corresponding labels in txt files. The images have original dimension of 100x100 pixels, and the different categories for labels are red blood cell, ring, schizont and trophozoite. <br> <br> The data was preprocessed by creating a csv file containing unique image id and the  encoded categorical target. The encoded categorical targets are as follows: 'red blood cell'=0, 'ring'=1, 'schizont'=2, 'trophozoite'=3. The original data was stratified split into 90-10% train-holdout and the train data was additionally split into 80-20% train-validation set. Hence the training set contains 6196 images, validation set contains 1550 images and holdout set contains 861 images.   
</div>

## Exploratory Data Analysis



## Modeling and Evaluation Metric

## Results & Analysis

## Conclusion

## Future Work

## References
1. https://www.cdc.gov/parasites/malaria/index.html
2. https://optuna.org/
