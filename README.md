# Gender Recognition Using Transformers

This repository includes code and dataset for building a gender recognition model. Two datasets have been used for this project: (1) [gender calssification dataset available on Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-user-gender-classification) and (2) [the PAN-18](https://zenodo.org/record/3746006) dataset. The Kaggle dataset has labeled the users with four classes, brand, female, male, and unknown. It provides one random tweet for each user, and the link to their profile images. Using the Twitter API Academic Researcher Account, we gathered more tweets for the female, male, and brand users of this dataset and provide the tweet and user IDs in the Dataset folder of this repository. The PAN-18 dataset has classified the users into two classes, female and male. It provides 100 tweets and 10 image contents posted by the user on Twitter for gender identification. 

We have built image and text-classification models for gender recognition using the Kaggle and the PAN-18 datasets, and then combined the models to get higher accuracy. For the Kaggle dataset, three transformer-based vision models, namely, [Vision Transformers (ViT)](https://huggingface.co/docs/transformers/model_doc/vit), [LeViT](https://huggingface.co/docs/transformers/model_doc/levit), and [Swin Transformer](https://huggingface.co/docs/transformers/model_doc/swin) were fine tuned to classify the useres based on their profile image. Moreover, the tweets of each user were divided into ten sets of ten tweets. The tweets of each set were concatenated. Three transformer-based text models, namely, [Bidirectional Encoder Representations from Transforers (BERT)](https://huggingface.co/docs/transformers/model_doc/bert), [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta), and [ELECTRA](https://huggingface.co/docs/transformers/model_doc/electra), were fine-tuned to learn the gender of each set. Finally, the result of the ten different sets were combined through two stacked neural network layers. Eventually the image and text classification models were combined through a stacked neural network to produce the final result.

For the PAN-18 dataset, the same vision models were used to build an image classification model. First, the ten different images of the users were combined, and for each user, ten new images were produced using the original images. The new images were fed into the transformer vision models to fine-tune them for gender classification. The code provided in "concatenate_images.py" regenerates the images used for each user. Then the gender recognition result of the ten newly generated images are combined useng two stacked feed-forward layers. For the text classification model, the same method and the same model explained for the Kaggle dataset was used. Eventually the image and text classification models were combined using stacked neural network layers.

The code for building our model, and producing our result is available in this repository. For the Kaggle dataset, first hydrate the tweets. Then download this repository and place it beside the hydrated tweets file. Executing any of the files will build the corresponding model. For the PAN-18 dataset, download the dataset and unzip the file. download this repository and place it beside the dataset folder. Execute the "concatenate_images.py" file to produce the images. By executing the code for other models, the models will be built. 

To execute the codes of this repository, the transformer and the PyTorch-Lightning packages need to be installed.

    pip install -q transformers
    pip install transformers pytorch-lightning --quiet

The code is also available in [this colab notebook](https://colab.research.google.com/drive/1e4YCl5qv-siLN2csaPpQdwn9gqgjGbCn).
