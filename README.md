# vascEmbeddings
Generated vascaulature embeddings for retina biomarker discovery

This repository contains all code used for generating the results described in:

Giancardo, L., Roberts, K. & Zhao, Z. Jan 1 2017 Fetal, Infant and Ophthalmic Medical Image Analysis - International Workshop, FIFI 2017 and 4th International Workshop, OMIA 2017 Held in Conjunction with MICCAI 2017, Proceedings. 


## Generating the Embeddings

The network weights file (test6/test6_best_weights.h5) need to be downloaded at the following address:
[https://drive.google.com/file/d/1UzAVgJKIsVFqQPfg-wk7d4yGW5k1gbCJ/view?usp=sharing](https://drive.google.com/file/d/1UzAVgJKIsVFqQPfg-wk7d4yGW5k1gbCJ/view?usp=sharing)
and placed in the test6 directory.

The following file shows how to generate the embedding from a test image on the 'data' folder:
[testEmbedding.py](testEmbedding.py)

It also include  a visual ouput of the vasculature segmentation used to drive the embedding. 

N.B. The function generating the embeddings reloads on the fly the model see the generateEncoding. An example of a more efficient way of generating the encodings for large dataset is in the generateEncoding function of [transfLearning.py](src/transfLearning.py)

## Requirements
This code requires Python 2.x 
 (although it can probably easily ported to 3.x)
 
 These are the packages that need to be installed
 ~~~
 conda install python=2.7 tensorflow-gpu keras pandas scikit-learn scikit-image matplotlib seaborn opencv ipython jupyter 
 ~~~
 
 The code was tested using the coda environment in this file:  [conda_env.yml](conda_env.yml)
 
 it can be replicated by running:
 ~~~
 conda create --name environmentName --file conda_env.yml
 ~~~
 
 