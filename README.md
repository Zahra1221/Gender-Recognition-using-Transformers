# Gender Recognition Using Transformers

This repository includes code and dataset for building a gender recognition model. Two datasets have been used for this project: (1) [gender calssification dataset available on Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-user-gender-classification) and (2) [the PAN-18](https://zenodo.org/record/3746006) dataset. The Kaggle dataset has labeled the users with four classes, brand, female, male, and unknown. It provides one random tweet for each user, and the link to their profile images. Using the Twitter API Academic Researcher Account, we gathered more tweets for the female, male, and brand users of this dataset and provide the tweet and user IDs in the Dataset folder of this repository. The PAN-18 dataset has classified the users into two classes, female and male. It provides 100 tweets and 10 image contents posted by the user on Twitter for gender identification. 

We have built image and text-classification models for gender recognition using the Kaggle and the PAN-18 datasets, and then combined the models to get higher accuracy for each dataset. For the Kaggle dataset, three transformer-based vision models, namely, [Vision Transformers (ViT)](https://huggingface.co/docs/transformers/model_doc/vit), [LeViT](https://huggingface.co/docs/transformers/model_doc/levit), and [Swin Transformer](https://huggingface.co/docs/transformers/model_doc/swin) were fine tuned to classify the useres based on their profile image. Moreover, the tweets of each user were divided into ten sets of ten tweets. The tweets of each set were concatenated. Three transformer-based text models, namely, [Bidirectional Encoder Representations from Transforers (BERT)](https://huggingface.co/docs/transformers/model_doc/bert), [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta), and [ELECTRA](https://huggingface.co/docs/transformers/model_doc/electra), were fine-tuned to learn the gender of each set. Finally, the result of the ten different sets were combined through two stacked neural network layers. Eventually the image and text classification models were combined through a stacked neural network to produce the final result.

For the PAN-18 dataset, the same vision models were used to build an image classification model. First, the ten different images of the users were combined, and for each user, ten new images were generated using the original images. The code provided in *"concatenate_images.py"* of this repository produces the combined images. 
The combined images were fed into the transformer vision models to fine-tune them for gender classification. Then the gender recognition result of the ten newly generated images are combined useng two stacked feed-forward layers. For the text classification model, the same method and the same model explained for the Kaggle dataset was used. Eventually the image and text classification models for the PAN-18 dataset were combined using two stacked neural network layers.

The code for building our model, and producing our result is available in this repository. For the Kaggle dataset, first download and unzip this repository. Then, hydrate the tweets and add the text and the link to users' profile images to the train.csv, validation.csv, and test.csv files, in the *"Dataset"* directory of this repository, under new columns named "Text" and "ProfileImage". Then execute the *"KaggleImages.py"* file in this repository. The images will automatically appear in new folders named kaggle_image_train, kaggle_image_validation, and kaggle_image_test. Now you are ready to execute the files in the *"Kaggle"* directory of this repository. 

For the PAN-18 dataset, first download the PAN-18 train and test datasets and unzip them. Then download this repository, place it beside this PAN-18 train and test datasets, and unzip it. Execute the *"concatenate_images.py"* file of this repository to produce the combined images. The combined images will automatically appear in two new folders, image_train and image_test. Now you are ready to execute the files in the *"PAN-18"* directory of this repository.

To execute the codes in this repository install these two packages:

    pip install -q transformers
    pip install transformers pytorch-lightning --quiet

The codes are also dependent to the following libraries:

    os
    shutil
    urllib
    cv2
    PIL
    pandas
    numpy

The code is also available in [this colab notebook](https://colab.research.google.com/drive/1e4YCl5qv-siLN2csaPpQdwn9gqgjGbCn).

If you use our code or dataset in your work, please kindly cite our manuscript:

[Z Movahedi Nia, A Ahmadi, B Mellado, J Wu, J Orbinski, A Asgary, J Kong, Twitter-based gender recognition using transformers, AIMS, Mathematical Biosciences and Engineering, 2023;20(9):15962-81, doi: 10.3934/mbe.2023711.](https://doi.org/10.3934/mbe.2023711)
