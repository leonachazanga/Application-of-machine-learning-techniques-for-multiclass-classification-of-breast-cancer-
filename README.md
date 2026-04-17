# Application-of-machine-learning-techniques-for-multiclass-classification-of-breast-cancer-
A deep learning project focused on multiclass classification of breast cancer histopathology images using the BreakHis dataset. This project applies transfer learning with ResNet-50, DenseNet121, and EfficientNet-B0, along with preprocessing, class balancing, and Grad-CAM to improve classification performance and interpretability.


# Introduction
Breast cancer is the most diagnosed cancer and leading cause of cancer death among women worldwide. It is comprised of a heterogeneous group of tumors with variations in clinical presentation, morphology, biological behavior, molecular features and response to therapy (Rakha, 2022). The breast cancer incidence and death rates have continuously increased over the past three decades with about 2.3 million new cases worldwide according to GLOBOCAN 2020 data (Łukasiewic et al 2021). 
The pathological and morphological classification of breast cancer is important as it provides diagnostic and prognostic information. Histomorphological classification plays a pivotal role in breast cancer diagnosis as it provided the foundation for all the other classification systems (Rakha et al, 2022). 
Artificial intelligence (AI), machine learning (ML) and deep learning (DL) help improve the ability of histopathologists to make more accurate and reproducible diagnoses (Dur Karasayar et al, 2025). AI-powered image analysis contributes to more precise staging, treatment planning, and reduced evaluation time. 


# Problem Statement
Breast cancer is characterized by the presence of multiple subtypes of tumors which may be difficult to distinguish due to the presence of high intra-class variability and inter-class similarity in histopathological images which may be difficult for pathologists to identify. Therefore, there is a need to develop AI based systems which can efficiently and accurately classify breast cancer tumor types and improve diagnostic accuracy and reduce time taken to process histopathology images. 
# Objectives
* To develop a machine learning model for multiclass classification of breast cancer subtypes using breast histopathology images
* To evaluate the model performance
* To use Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize image sections influencing model classification decisions 

# Dataset
BreakHis Dataset containing a total of 7909 breast histopathology images was used. The dataset comprises of 2 categories, Benign and Malignant images. 
Benign (2480 images)-Adenosis (A), Fibroadenoma (F), Phyllodes Tumor (PT), Tubular Adenoma (TA)
Malignant (5429 images)- Ductal Carcinoma (DC), Lobular Carcinoma (LC), Mucinous Carcinoma (MC), Papillary Carcinoma (PC)
The 100X magnification dataset with 2081 was used for this project because it provides enough morphological detail while maintaining computational efficiency. 
Dataset link:
http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz


# Dataset Features and Relevance
High-resolution histopathology images capturing cellular structures
Eight tumor classes for multiclass classification
Standardized microscopy settings
Real clinical images from 84 patients with confirmed diagnoses
These characteristics allow the dataset to be used in AI models which can assist in clinical diagnosis. 
Dataset Validation and Data Quality
Dataset labeled by pathologists with confirmed diagnoses
Introduced in a peer-reviewed scientific publication
Represents real clinical conditions
Preprocessing steps applied to ensure data quality
#Data Preprocessing and Transformations
The following preprocessing steps were applied:
Images were checked for correct formats
Converted to RGB format
Standardized image dimensions
Image resizing: 240 × 240
Normalization- Mean(0.485, 0.456, 0.406), Std (0.229, 0.224, 0.225)
Data Splitting- Training set (70%), Validation set (15%), Test set (15%)
Stratified sampling- to ensure proportional representation across classes. 
Class balance- WeightedRandomSampler used as dataset is imbalanced (approx. 70% malignant).
Models used- ResNet-50, DenseNet121, EfficientNet-B0
All models use- Transfer Learning (ImageNet pretrained weights), modified final layer for 8-class classification

# Model Descriptions
ResNet-50- uses residual connections to reduce vanishing gradients
DenseNet121- Dense connections improve feature reuse
EfficientNet-B0- Balanced scaling of depth, width, and resolution
# Model Selection
* Models were evaluated using accuracy, precision, recall, F1-score, ROC-AUC, Loss
* Fine-Tuning Strategy
* Increased image size to 224 × 224
* Batch size: 16
* Epochs- increased from 10 to 20
* Learning rate: 1e-4
* Early stopping- patience = 5
* Backbone partially unfrozen
Improved learning efficiency and model accuracy 

# Model Evaluation and Interpretability
* Confusion Matrix- shows strong classification performance with minor confusion between similar tumor types 
* ROC Curves- high AUC values indicate strong class separability 
* Learning Curves- showed stable training and reduced loss over epochs 
* Grad-CAM was used to highlight important regions in histopathology images showing tumor-relevant areas to ensure the model is learning meaningful features

# Results #
# Baseline model (Code part 1)
In the Code Part 1, the baseline model was developed using a transfer learning approach with pretrained convolutional neural networks. The aim was to establish an initial performance benchmark before applying fine-tuning and advanced optimization techniques. Pretrained architectures (ResNet-50, DenseNet121, and EfficientNet-B0) were initialized with ImageNet weights. The final classification layer of each model was replaced with a new fully connected layer consisting of 8 output neurons, corresponding to the eight breast cancer subtypes. The pretrained layers were kept frozen with only the final classification layer was trained. This allowed the model to retain previously learned general image features while adapting to the multiclassification task. The baseline results showed moderate performance, with accuracy values ranging between 63%–67% and F1-scores of about 64%. This indicated that while the models were learning meaningful patterns, their performance was limited due to restricted training (frozen layers) and lack of fine-tuning. The baseline model served as a reference point, highlighting the need for further optimization through fine-tuning, improved preprocessing, and enhanced training strategies, which were implemented in the “Part 2 code” stage of the project.
# Tuned model (Code Part 2)
During validation, ResNet-50 achieved the highest accuracy (0.7051) and F1-score (0.6651), indicating the best overall performance among the three models. DenseNet121 showed moderate performance (accuracy 0.6795, F1-score 0.6419), performing slightly lower than ResNet-50 but still reasonably strong. EfficientNet-B0 had the lowest performance (accuracy 0.6410, F1-score 0.6057), suggesting it was less effective for this classification of the breast cancer images. 
# Final test set comparison
All three models demonstrated strong performance, with accuracies ranging from approximately 89% to 91% and very high AUC values of approx. 0.99, indicating excellent ability to distinguish between breast cancer subtypes. DenseNet121 provided the most balanced performance, achieving the highest F1-score (0.8999) and precision (0.9081), which suggests it is the most reliable model across all classes with fewer false positives. ResNet-50 achieved the highest overall accuracy (0.9073), recall (0.9052), and AUC (0.9949), indicating it performs best in correctly identifying true cases and has the strongest class separability. EfficientNet-B0 shows slightly lower performance across all metrics, with reduced precision (0.8813) and F1-score (0.8845). Overall, DenseNet121 is the best choice for balanced performance, while ResNet-50 offers the highest predictive power, and EfficientNet-B0 performs slightly less effectively compared to the other models. For clinical purposes, ResNet50 would be the best model as it performed best in correctly identifying true cases

# Limitations
* Class imbalance
* Few epochs due to hardware limitations
* Limited dataset size
* Partial fine-tuning

# Conclusion #
* Successfully developed a multiclass breast cancer classification model
* Achieved approximately 90% accuracy with strong F1-scores after fine-tuning
* DenseNet121 performed best overall, ResNet-50 achieved highest accuracy/AUC
* Grad-CAM confirmed model focuses on relevant tumor regions
* Demonstrates strong potential of AI for cancer histopathology image processing 

# Future Work
* Increase number of epochs to improve generalization
* Expand dataset
* Apply advanced augmentation techniques
* Deploy model for clinical use
* Technologies Used
* Python (PyTorch, NumPy, Pandas, Matplotlib, scikit-learn)


