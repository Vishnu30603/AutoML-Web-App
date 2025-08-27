# AutoML Web Application  

An **end-to-end AutoML web app** built using **Streamlit** and **PyCaret**, enabling users to quickly upload datasets, perform exploratory data analysis (EDA), build machine learning models (Classification, Regression, or Clustering), interpret results, and download trained models — all without writing a single line of code.  

---

## Features  

### Upload Dataset  
- Upload `.csv` files  
- Automatic cleaning & categorical column conversion  

### Exploratory Data Analysis (EDA)  
- Dataset overview (shape, data types, missing values, summary stats)  
- Distribution plots for numerical & categorical features  
- Boxplots for outlier detection  
- Correlation heatmap  

### Model Building  
- Choose **Classification**, **Regression**, or **Clustering**  
- Auto train & compare multiple models using **PyCaret**  
- Automatic preprocessing (optional), imbalance handling, and outlier removal  
- Cross-validation with adjustable folds  

### Clustering Support  
- KMeans & Agglomerative Clustering  
- PCA-based 2D visualization of clusters  
- Silhouette score for quality evaluation  

### Model Interpretability  
- Feature importance plots  
- **SHAP** summary plots for explainability  
- Confusion Matrix, ROC/AUC (Classification)  
- Residuals Plot (Regression)  

### Model Evaluation  
- Leaderboard comparison of models  
- Test set performance metrics  

### Download Trained Models  
- Export trained models (`.pkl`)  
- Export cleaned dataset for future use  

---
