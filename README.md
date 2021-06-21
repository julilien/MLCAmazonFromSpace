# Multi-label Classification: Satellite Photos of the Amazon Rainforest

This is a simple mutli-label classification implementation on the [Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space) dataset, which consists of satellite images from the Amazon region. The implementation is based on the [blog post by Jason Brownlee](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/). We ported his code to Tensorflow 2 and slightly augmented it, but, however, the credit is entirely his.

## Requirements

### Packages

To execute the code, just set-up a Python 3.* environment and execute the following command:
```
pip -r install requirements.txt
```

### Dataset

In order to download the aforementioned dataset, one needs to sign in to Kaggle and download `train-jpg.tar` and `train_v2.csv` [here](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data). The training files need to extracted and stored to a subdirectory `data/`.

## Run the code

First, you need to prepare resized training and test images for an improved run-time. To this end, run
```
python3 prepare_data.py
```

Afterwards, you can perform a cross-validation (default is 3-fold) on the training images by executing
```
python3 train_cv.py
```

For each fold, it trains a neural network using a VGG16-based encoder (only the last convolutional block is unfreezed) pretrained on ImageNet with a simple classification head on the fold's training set. The model predictions for the fold images will be written to `predictions_{fold_idx}.npy`, whereas the corresponding filenames are stored in `filenames_{fold_idx}.pkl`.