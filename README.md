# Brain_Tumor_Classification

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
