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
- Dataset Description	
- Exploratory Data Analysis	
- Data Preprocessing	
- Modeling	
- Results & Analysis	
- Conclusion
- Future Work	
- References

## Introduction

<div style="text-align: justify">
Malaria is a mosquito-borne disease caused by a parasite. About 2,000 cases of malaria are diagnosed in the United States each year.<sup>[1]</sup> People with malaria often experience fever, chills, and flu-like illness. Left untreated, they may develop severe complications and die. For this project a Multi-Layer Perceptron (MLP) was used to classify blood cells into 4 different categories: red blood cell, ring, schizont and trophozoite. An image labelled as red blood cell is healthy and the rest of the labels indicate some stage of malaria infection. <br>If a model can successfully identify these types of cells from images taken with a microscope, this would allow to automate a very time-consuming testing process, leaving human doctors with more time to treat the actual disease. Furthermore, an early malaria detection can save lives!

</div>

## References
1. https://www.cdc.gov/parasites/malaria/index.html
2. 