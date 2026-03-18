#  Skin Lesion Classifier

##  Overview
This project implements a deep learning model to classify skin lesions from images. The goal is to assist in the early detection of potential skin conditions by automatically analyzing visual features, and classifing them into the correesponding class.

The model takes an input image of a skin lesion and outputs a predicted class among 8 categories.

---

##  Demo
Try the live demo here : [Live Demo](https://huggingface.co/spaces/HassanAiolio/skin-lesion-classifier)

---

##  Dataset
- **Dataset**: ISIC 2019 Skin Lesion Classification  
- **Number of classes**: 8  
- **Total images**: 25,331  

### Class distribution
- MEL: Melanoma — 4,522
- NV: Melanocytic Nevus — 12,875
- BCC: Basal Cell Carcinoma — 3,323
- BKL: Benign Keratosis — 2,624
- AK: Actinic Keratosis — 867
- SCC: Squamous Cell Carcinoma — 628
- VASC: Vascular Lesion — 253
- DF: Dermatofibroma — 239 

The dataset is highly imbalanced, with some classes significantly underrepresented.

To address this, a **weighted cross-entropy loss** was used during training to give more importance to minority classes.

---

##  Model
- **Architecture**: EfficientNet-B0 (pretrained)  
- **Framework**: PyTorch  
- **Modification**: Final classification layer adapted to 8 output classes  

A learning rate scheduler was used to improve convergence during training.

---

##  Training
- **Loss function**: Weighted Cross-Entropy  
- **Optimizer**: Adam  
- **Epochs**: 50  
- **Learning rate scheduler**: Enabled  

Training was done in two phases:
1. Backbone frozen — only the classifier head trained for the first 5 epochs
2. Full fine-tuning — entire network trained at a lower learning rate (1e-4)

The model achieves approximately:

 **84% validation accuracy**

---

## Retraining the Model
To retrain the model, it is recommended to use Kaggle due to dataset availability and compute resources.

Dataset path on Kaggle:
[ISIC 2019 on Kaggle](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification)

Steps:
1. Create a Kaggle notebook
2. Attach the dataset above
3. Upload the dataset.py, model.py and train.py files
4. Run it and make sure you are on the GPU device

---

##  Run Locally

```bash
pip install -r requirements.txt
python app.py