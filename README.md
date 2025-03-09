# Breast Cancer Prediction 🎗️
Predicting the likelihood of breast cancer using Machine Learning models.

## 📌 Project Overview
Breast cancer is one of the most common cancers among women worldwide. Early detection and accurate classification of breast tumors can help in better treatment and patient care. This project uses **the Wisconsin Diagnostic Breast Cancer (WDBC) dataset** to build a predictive model for **benign vs. malignant tumor classification**.

## 📊 Dataset
- The dataset used in this project is **Wisconsin Breast Cancer Dataset (WDBC)**.
- It contains **30 features** extracted from cell nucleus images.
- The target variable (`diagnosis`) has two classes:
  - **M (Malignant - Cancerous)**
  - **B (Benign - Non-cancerous)**

##  Machine Learning Models Used
The following models were trained and evaluated:
1. **Support Vector Machine (SVM)**
2. **Random Forest Classifier**
3. **Gradient Boosting Classifier**
4. **K-Nearest Neighbors (KNN)**
5. **Logistic Regression**
6. **Multi-Layer Perceptron (Neural Network)**
7. **Naive Bayes (Gaussian & Multinomial)**

## 🔍 Performance Metrics
- **Accuracy**
- **Precision**
- **Recall (Sensitivity)**
- **F1-Score**
- **Confusion Matrix**
- **ROC-AUC Curve**

📈 Data Visualization
The project includes exploratory data analysis (EDA) using Matplotlib & Seaborn, including:
- Feature distributions
- Correlation heatmaps
- Boxplots to identify outliers
- Histograms for class distributions

 ** Key Takeaways: **
- The Random Forest Classifier outperformed other models with highest accuracy & recall.
- Feature standardization significantly improved the SVM and MLP model performances.
- The dataset showed high class imbalance, which required handling with weighted loss functions.


📜 License
This project is licensed under the MIT License.
