# CTS-Hackathon-NPN-Program

## Dataset: ## 
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## Pre-processing: ##
CLAHE (Contrast Limited Adaptive Histogram Equalization) is a powerful preprocessing technique that enhances the contrast of an image, especially useful for medical images like brain MRI scans. It adjusts the histogram of the image in small regions to improve local contrast while avoiding noise amplification, which is particularly helpful in enhancing features like tumors.

## Model: ##
Combining VGG16 and InceptionV3 in a hybrid model for brain tumor classification and freezing the initial 10 layers of both models to leverage their pre-trained feature extraction capabilities, while letting the later layers learn from your dataset. This approach is commonly used to fine-tune the models for specific tasks like classification of different types of brain tumors (meningioma, pituitary, glioma, no tumor).

### Note: Save the model as pickle(.pkl) file (Refer .ipynb file) for classification.
