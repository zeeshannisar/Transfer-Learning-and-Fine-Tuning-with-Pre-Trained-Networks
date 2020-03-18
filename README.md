# Transfer Learning and Fine Tuning in Keras with State of the Art Pre Trained Networks:
This Repository contains the detailed description and implementation of Transfer-Learning and Fine-Tuning solution for Image Classification Problems in Computer Vision domain.



## Table of Contents
   + [Transfer Learning](#transfer-learning)
   + [Transfer Learning Strategies](#transfer-learning-strategies)
   + [Transfer Learning Process](#transfer-learning-process)
   + [Datasets and Code Implementations](#datasets-and-code-implementations)

### Transfer Learning:
Transfer learning generally refers to a process where a model trained on one problem is used in some way on a second related problem. It is a popular method in computer vision domain because it allows us to build accurate models in a timesaving way (Rawat & Wang 2017). With Transfer learning, instead of starting the learning from scratch, we start from patterns that have already been learnt while solving a different but related problem. This way we leverage previous learnings and avoid learning from scratch.

### Transfer Learning Strategies:
When you’re repurposing a pre-trained model for your own needs, you start by removing the original classifier, then you add a new classifier that fits your purposes, and finally you have to fine-tune your model according to one of three listed strategies. These strategies are also detailed in [Figure: 1](#schematic-diagram-of-transfer-learning-strategies)

#### 1. Train the entire model:
In this case, you use the architecture of the pre-trained model and train it according to your dataset. You’re learning the model from scratch, so you’ll need a large dataset (and a lot of computational power).

#### 2. Train some layers and leave the others frozen:
As you remember, lower layers refer to general features (problem independent), while higher layers refer to specific features (problem dependent). Here, we play with that dichotomy by choosing how much we want to adjust the weights of the network (a frozen layer does not change during training). Usually, if you’ve a small dataset and a large number of parameters, you’ll leave more layers frozen to avoid overfitting. By contrast, if the dataset is large and the number of parameters is small, you can improve your model by training more layers to the new task since overfitting is not an issue.

#### Freeze the convolutional base.
This case corresponds to an extreme situation of the train/freeze trade-off. The main idea is to keep the convolutional base in its original form and then use its outputs to feed the classifier. You’re using the pre-trained model as a fixed feature extraction mechanism, which can be useful if you’re short on computational power, your dataset is small, and/or pre-trained model solves a problem very similar to the one you want to solve.

#### Schematic Diagram of Transfer Learning Strategies: 
<p align="center">
    <img src="https://github.com/zeeshannisar/Transfer-Learning-and-Fine-Tuning-with-Pre-Trained-Networks/blob/master/ReadMe%20Images/trasnfer%20learning%20strategies.png">
    <br>
    <em> Figure:1 Transfer Learning Strategies in a Schematic way. </em>
</p>

Unlike **Strategy 3**, whose application is straightforward, **Strategy 1** and **Strategy 2** require you to be careful with the learning rate used in the convolutional part. The learning rate is a hyper-parameter that controls how much you adjust the weights of your network. When you’re using a pre-trained model based on CNN, it’s smart to use a small **learning rate** because high learning rates increase the risk of losing previous knowledge. Assuming that the pre-trained model has been well trained, which is a fair assumption, keeping a small learning rate will ensure that you don’t distort the CNN weights too soon and too much.

### Transfer Learning Process:

#### 1. Select a pre-trained model:
From the wide range of pre-trained models that are available, you pick one that looks suitable for your problem. For example, if you’re using Keras, you immediately have access to a set of models, such as VGG (Simonyan & Zisserman 2014), InceptionV3 (Szegedy et al. 2015), and ResNet5 (He et al. 2015). [Here](https://keras.io/applications/) you can see all the models available on Keras.

#### 2. Classify your problem according to the Size-Similarity Matrix:
In [Figure: 2](#size-similarity-matrix-and-decision-map) you have ‘The Matrix’ that controls your choices. This matrix classifies your computer vision problem considering the size of your dataset and its similarity to the dataset in which your pre-trained model was trained. As a rule of thumb, consider that your dataset is small if it has less than 1000 images per class. Regarding dataset similarity, let common sense prevail. For example, if your task is to identify cats and dogs, ImageNet would be a similar dataset because it has images of cats and dogs. However, if your task is to identify cancer cells, ImageNet can’t be considered a similar dataset.

#### 3. Fine-tune your model:
Here you can use the Size-Similarity Matrix to guide your choice and then refer to the three options we mentioned before about repurposing a pre-trained model. [Figure: 2](#size-similarity-matrix-and-decision-map) provides a visual summary of the text that follows.

##### a.) Quadrant 1
Large dataset, but different from the pre-trained model’s dataset. This situation will lead you to Strategy 1. Since you have a large dataset, you’re able to train a model from scratch and do whatever you want. Despite the dataset dissimilarity, in practice, it can still be useful to initialise your model from a pre-trained model, using its architecture and weights.

##### b.) Quadrant 2.
Large dataset and similar to the pre-trained model’s dataset. Here you’re in la-la land. Any option works. Probably, the most efficient option is Strategy 2. Since we have a large dataset, overfitting shouldn’t be an issue, so we can learn as much as we want. However, since the datasets are similar, we can save ourselves from a huge training effort by leveraging previous knowledge. Therefore, it should be enough to train the classifier and the top layers of the convolutional base.

##### c.) Quadrant 3.
Small dataset and different from the pre-trained model’s dataset. This is the 2–7 off-suit hand of computer vision problems. Everything is against you. If complaining is not an option, the only hope you have is Strategy 2. It will be hard to find a balance between the number of layers to train and freeze. If you go to deep your model can overfit, if you stay in the shallow end of your model you won’t learn anything useful. Probably, you’ll need to go deeper than in Quadrant 2 and you’ll need to consider data augmentation techniques (a nice summary on data augmentation techniques is provided here).

##### d.) Quadrant 4.
Small dataset, but similar to the pre-trained model’s dataset. I asked Master Yoda about this one he told me that ‘be the best option, Strategy 3 should’. I don’t know about you, but I don’t underestimate the Force. Accordingly, go for Strategy 3. You just need to remove the last fully-connected layer (output layer), run the pre-trained model as a fixed feature extractor, and then use the resulting features to train a new classifier.

#### Size-Similarity Matrix and Decision Map:
<p align="center">
    <img src="https://github.com/zeeshannisar/Transfer-Learning-and-Fine-Tuning-with-Pre-Trained-Networks/blob/master/ReadMe%20Images/trasnfer%20learning%20flow.png">
    <br>
    <em> Figure:2 Size-Similarity matrix (left) and decision map for fine-tuning pre-trained models (right). </em>
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
[Code: Google Colab Notebook](https://github.com/zeeshannisar/Transfer-Learning-and-Fine-Tuning-with-Pre-Trained-Networks/blob/master/Fine%20Tuning%20with%20VGG16/Fine%20tuning%20with%20Pretrained%20VGG16%20for%20Tuberculosis%20Classification.ipynb)

##### Transfer Learning with VGG19:
[Code: Google Colab Notebook](https://github.com/zeeshannisar/Transfer-Learning-and-Fine-Tuning-with-Pre-Trained-Networks/blob/master/Fine%20Tuning%20with%20VGG19/Fine%20tuning%20with%20Pretrained%20VGG19%20for%20Tuberculosis%20Classification%20.ipynb)

##### Transfer Learning with AttentionBased-VGG16:
[Code: Google Colab Notebook](https://github.com/zeeshannisar/Transfer-Learning-and-Fine-Tuning-with-Pre-Trained-Networks/blob/master/Fine%20Tuning%20with%20AttentionBasedVGG16/Fine%20Tuning%20with%20Attention%20Based%20pre-Trained%20VGG16%20for%20Tuberculosis%20Classification.ipynb)

##### Transfer Learning with Res-Net50:
[Code: Google Colab Notebook](https://github.com/zeeshannisar/Transfer-Learning-and-Fine-Tuning-with-Pre-Trained-Networks/blob/master/Fine%20Tuning%20with%20Res-Net50/Fine%20Tuning%20with%20Pre-Trained%20Res-Net50%20for%20Melanoma(Skin%20Cancer)%20Detection.ipynb)

