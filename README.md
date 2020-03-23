# Transfer Learning and Fine Tuning in Keras with State of the Art Pre Trained Networks:
This Repository contains the detailed description and implementation of Transfer-Learning and Fine-Tuning solution for Image Classification Problems in Computer Vision domain.

## Table of Contents
   + [Transfer Learning](#transfer-learning)
   + [Transfer Learning Strategies](#transfer-learning-strategies)
   + [Transfer Learning Process](#transfer-learning-process)
   + [Datasets and Code Implementations](#datasets-and-code-implementations)

### Transfer Learning:
Transfer learning refers to a process where a model trained on one problem is used in some way on a second related problem. It is a popular method in Computer-Vision domain because it allows us to build effecient models in a time-saving way (Rawat & Wang 2017). With Transfer learning, instead of learning the model from scratch, we start from patterns that have already been learnt while solving a different but related problem. This way we leverage previous learnings and avoid learning from scratch.

### Transfer Learning Strategies:
When we’re reusing a pre-trained model for our own needs, we start by removing the original classifier, then we add a new classifier that fits our purposes, and finally we have to fine-tune our model according to one of the three listed strategies.

<p align="center">
    <img src="https://github.com/zeeshannisar/Transfer-Learning-and-Fine-Tuning-with-Pre-Trained-Networks/blob/master/ReadMe%20Images/trasnfer%20learning%20strategies.png">
    <br>
    <em> Figure:1 Transfer Learning Strategies. </em>
</p>

### Transfer Learning Process:

#### 1. Select a pre-trained model:
From the wide range of available pre-trained models at [Here](https://keras.io/applications/), we have to pick one that looks suitable for our problem.
#### 2. Classify the problem and Fine-Tune the Model according to the Size-Similarity Matrix:
In [Figure: 2](#size-similarity-matrix-and-decision-map) we have **The Size-Similarity Matrix** that controls our choices. This matrix classifies the Computer-Vision problem considering the size of the dataset and its similarity to the dataset in which pre-trained model is to be trained.

   + Large data and diff from pretrained dataset (train entire model)
   + Large data and same as pretrained dataset (train some layers and freeze others)
   + Small data and diff from pretrained dataset (train some layers and freeze others)
   + Small data and same as pretrained dataset (freeze the conv base)  

#### Size-Similarity Matrix and Decision Map:
<p align="center">
    <img src="https://github.com/zeeshannisar/Transfer-Learning-and-Fine-Tuning-with-Pre-Trained-Networks/blob/master/ReadMe%20Images/trasnfer%20learning%20flow.png">
    <br>
    <em> Figure: 2 Size-Similarity matrix (left) and Decision-Map to Fine-Tune Pre-trained Models (right). </em>
</p>

### Datasets and Code Implementations:
I have used the following pretrained networks for Transfer-Learning for Tuberculosis Classification and Skin Cancer Detection tasks.

+ [Fine Tuning with VGG16](#fine-tuning-with-vgg16)    
+ [Fine Tuning with VGG19](#fine-tuning-with-vgg19)
+ [Fine Tuning with AttentionBased-VGG16](#fine-tuning-with-attentionbased-vgg16)
+ [Fine Tuning with Res-Net50](#fine-tuning-with-resnet-50)

#### Datasets:
The original datasets are publicaly available at [Tuberculosis Dataset](https://lhncbc.nlm.nih.gov/publication/pub9931) and [Skin Cancer Datset](https://www.kaggle.com/drscarlat/melanoma) and can be also be requested at zshnnisar@gmail.com to get the same accuracy results. 


#### Implementations:

##### Transfer Learning with VGG16:
[Code: Google Colab Notebook](https://github.com/zeeshannisar/Transfer-Learning-and-Fine-Tuning-with-Pre-Trained-Networks/blob/master/Transfer%20Learning%20with%20VGG16/Fine%20tuning%20with%20Pretrained%20VGG16%20for%20Tuberculosis%20Classification.ipynb)

##### Transfer Learning with VGG19:
[Code: Google Colab Notebook](https://github.com/zeeshannisar/Transfer-Learning-and-Fine-Tuning-with-Pre-Trained-Networks/blob/master/Transfer%20Learning%20with%20VGG19/Fine%20tuning%20with%20Pretrained%20VGG19%20for%20Tuberculosis%20Classification%20.ipynb)

##### Transfer Learning with AttentionBased-VGG16:
[Code: Google Colab Notebook](https://github.com/zeeshannisar/Transfer-Learning-and-Fine-Tuning-with-Pre-Trained-Networks/blob/master/Transfer%20Learning%20with%20AttentionBasedVGG16/Fine%20Tuning%20with%20Attention%20Based%20pre-Trained%20VGG16%20for%20Tuberculosis%20Classification.ipynb)

##### Transfer Learning with Res-Net50:
[Code: Google Colab Notebook](https://github.com/zeeshannisar/Transfer-Learning-and-Fine-Tuning-with-Pre-Trained-Networks/blob/master/Transfer%20Learning%20with%20Res-Net50/Fine%20Tuning%20with%20Pre-Trained%20Res-Net50%20for%20Melanoma(Skin%20Cancer)%20Detection.ipynb)

