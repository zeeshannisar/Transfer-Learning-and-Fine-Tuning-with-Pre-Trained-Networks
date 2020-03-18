# Transfer Learning and Fine Tuning in Keras with State of the Art Pre Trained Networks:
This Repository contains the implementation of Transfer-Learning and Fine-Tuning solution for classification problems. Transfer learning generally refers to a process where a model trained on one problem is used in some way on a second related problem. It is a popular method in computer vision domain because it allows us to build accurate models in a timesaving way (Rawat & Wang 2017). With Transfer learning, instead of starting the learning from scratch, we start from patterns that have already been learnt while solving a different but related problem. This way we leverage previous learnings and avoid learning from scratch.

## Datasets and Pretrained Networks:

### Pretrained Networks:
I have used the following pretrained networks for Transfer-Learning for Tuberculosis Classification and Skin Cancer Detection tasks.

+ [Fine Tuning with VGG16](#fine-tuning-with-vgg16)    
+ [Fine Tuning with VGG19](#fine-tuning-with-vgg19)
+ [Fine Tuning with AttentionBased-VGG16](#fine-tuning-with-attentionbased-vgg16)
+ [Fine Tuning with Res-Net50](#fine-tuning-with-resnet-50)

### Datasets:
The original datasets are publicaly available at [Tuberculosis Dataset](https://lhncbc.nlm.nih.gov/publication/pub9931) and [Skin Cancer Datset](https://www.kaggle.com/drscarlat/melanoma) and can be also be requested at zshnnisar@gmail.com to get the same accuracy results. 


## Implementations:

### Fine Tuning with VGG16:
[Code: Google Colab Notebook](https://github.com/zeeshannisar/Transfer-Learning-and-Fine-Tuning-with-Pre-Trained-Networks/blob/master/Fine%20Tuning%20with%20VGG16/Fine%20tuning%20with%20Pretrained%20VGG16%20for%20Tuberculosis%20Classification.ipynb)

### Fine Tuning with VGG19:
[Code: Google Colab Notebook](https://github.com/zeeshannisar/Transfer-Learning-and-Fine-Tuning-with-Pre-Trained-Networks/blob/master/Fine%20Tuning%20with%20VGG19/Fine%20tuning%20with%20Pretrained%20VGG19%20for%20Tuberculosis%20Classification%20.ipynb)

### Fine Tuning with AttentionBased-VGG16:
[Code: Google Colab Notebook](https://github.com/zeeshannisar/Transfer-Learning-and-Fine-Tuning-with-Pre-Trained-Networks/blob/master/Fine%20Tuning%20with%20AttentionBasedVGG16/Fine%20Tuning%20with%20Attention%20Based%20pre-Trained%20VGG16%20for%20Tuberculosis%20Classification.ipynb)

### Fine Tuning with Res-Net50:
[Code: Google Colab Notebook](https://github.com/zeeshannisar/Transfer-Learning-and-Fine-Tuning-with-Pre-Trained-Networks/blob/master/Fine%20Tuning%20with%20Res-Net50/Fine%20Tuning%20with%20Pre-Trained%20Res-Net50%20for%20Melanoma(Skin%20Cancer)%20Detection.ipynb)

