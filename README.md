# Brain_Tumor_Classification

This web application performs automated brain tumor classification by processing MRI images (150×150×3) through a hybrid deep learning pipeline. It fuses features from pretrained VGG16 (capturing fine spatial details) and InceptionV3 (capturing multi-scale abstract patterns), concatenates the extracted features, and passes them through fully connected dense layers ending with a softmax layer to classify tumors into four categories. The app delivers highly accurate classifications/predictions in real-time, providing an interactive and informative interface for clinicians/patients to upload scans and obtain  first-order diagnostic results efficiently.

## Dataset: ## 
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

## Pre-processing: ##
CLAHE (Contrast Limited Adaptive Histogram Equalization) is a powerful preprocessing technique that enhances the contrast of an image, especially useful for medical images like brain MRI scans. It adjusts the histogram of the image in small regions to improve local contrast while avoiding noise amplification, which is particularly helpful in enhancing features like tumors.

## Model: ##
Combining VGG16 and InceptionV3 in a hybrid model for brain tumor classification and freezing the initial 10 layers of both models to leverage their pre-trained feature extraction capabilities, while letting the later layers learn from your dataset. This approach is commonly used to fine-tune the models for specific tasks like classification of different types of brain tumors (meningioma, pituitary, glioma, no tumor).

### Note: Save the model as pickle(.pkl) file (Refer .ipynb file) for classification.
# Results:
```bash
41/41 ━━━━━━━━━━━━━━━━━━━━ 18s 291ms/step
Accuracy: 0.9862700228832952
Precision: 0.9863117466225336
Recall: 0.9862700228832952
F1 Score: 0.9862821416895429
Classification Report:
              precision    recall  f1-score   support

      glioma       0.99      0.98      0.99       300
  meningioma       0.97      0.97      0.97       306
     notumor       0.99      0.99      0.99       405
   pituitary       0.99      0.99      0.99       300

    accuracy                           0.99      1311
   macro avg       0.99      0.99      0.99      1311
weighted avg       0.99      0.99      0.99      1311

Confusion Matrix:
[[295   5   0   0]
 [  2 298   3   3]
 [  0   3 402   0]
 [  0   2   0 298]]
```

# Confusion Matrix
<img width="513" height="470" alt="image" src="https://github.com/user-attachments/assets/6afdb3bb-4318-45ec-8daa-450a4fefe3c3" />


# Classification
<img width="1570" height="750" alt="image" src="https://github.com/user-attachments/assets/6ac95667-3004-4d7e-ab94-c33e6c1e8bcb" />

