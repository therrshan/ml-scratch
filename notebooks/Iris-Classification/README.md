# Iris Classification Project

This folder contains a comprehensive machine learning analysis of the classic Iris dataset using our custom ML algorithms implemented from scratch.

## Dataset

The **Iris dataset** is a classic machine learning dataset containing measurements of 150 iris flowers from three different species:
- **Setosa** (50 samples)
- **Versicolor** (50 samples) 
- **Virginica** (50 samples)

### Features
- **sepal length (cm)**: Length of sepal in centimeters
- **sepal width (cm)**: Width of sepal in centimeters  
- **petal length (cm)**: Length of petal in centimeters
- **petal width (cm)**: Width of petal in centimeters

## Project Structure

```
Iris-Classification/
├── 01-Iris-EDA-Cleaning.ipynb    # Data exploration and preprocessing
├── 02-Iris-Modelling.ipynb       # ML algorithm applications
└── README.md                     # This file
```

## Data Processing Pipeline

### 1. Exploratory Data Analysis (`01-Iris-EDA-Cleaning.ipynb`)
- **Data Loading**: Load Iris dataset from sklearn
- **Missing Values**: Check and handle missing data
- **Data Distribution**: Analyze feature distributions by class
- **Correlation Analysis**: Examine feature relationships
- **Outlier Detection**: Identify and analyze outliers
- **Feature Engineering**: Create new features:
  - Aspect ratios (sepal, petal)
  - Area approximations (sepal, petal, total)
  - Perimeter approximations
  - Petal-to-sepal ratios
- **Data Scaling**: Apply StandardScaler for normalization
- **Train-Test Split**: 80-20 stratified split
- **Data Export**: Save processed data to `../../data/IrisProcessed/`

### 2. Machine Learning Modeling (`02-Iris-Modelling.ipynb`)

#### Supervised Learning Algorithms
1. **Logistic Regression** (One-vs-Rest multiclass)
2. **Neural Network** (Multi-layer with ReLU activation)
3. **Random Forest** (100 trees with feature importance)
4. **K-Nearest Neighbors** (k=5, Euclidean distance)
5. **Decision Tree** (Gini criterion)
6. **Naive Bayes** (Gaussian assumption)
7. **Support Vector Machine** (RBF kernel)

#### Unsupervised Learning Algorithms
1. **K-Means Clustering** (3 clusters)
2. **Principal Component Analysis** (2 components)

## Key Features

### Feature Engineering
The project includes comprehensive feature engineering:
- **8 engineered features** in addition to the original 4
- **Aspect ratios** for shape analysis
- **Area calculations** for size analysis
- **Perimeter approximations** for boundary analysis

### Model Evaluation
- **Accuracy, Precision, Recall, F1-Score**
- **Confusion matrices** for each model
- **Feature importance analysis** (Random Forest)
- **Model comparison** and ranking
- **Error analysis** for best performing model

### Visualization
- **Feature distributions** by class
- **Correlation heatmaps**
- **Training history plots** (Neural Network)
- **Feature importance plots** (Random Forest)
- **PCA visualizations**
- **Cluster analysis plots**

## Expected Results

The Iris dataset is well-separated, so most models should achieve:
- **High accuracy** (>90% for most algorithms)
- **Good class separation** in PCA space
- **Clear feature importance** patterns
- **Consistent performance** across different algorithms

## Usage

1. **Run EDA notebook first**: `01-Iris-EDA-Cleaning.ipynb`
2. **Run modeling notebook**: `02-Iris-Modelling.ipynb`
3. **Analyze results** and compare model performances

## Data Files

- **Raw data**: `../../data/IrisData.csv`
- **Processed data**: `../../data/IrisProcessed/`
  - `X_train.csv`, `X_test.csv`
  - `y_train.csv`, `y_test.csv`
  - `preprocessing_info.pkl`

## Technical Details

- **Total features**: 12 (4 original + 8 engineered)
- **Training samples**: 120 (80%)
- **Test samples**: 30 (20%)
- **Classes**: 3 (balanced)
- **Scaling**: StandardScaler applied
- **Cross-validation**: Stratified train-test split

This project demonstrates the application of all custom ML algorithms to a classic, well-understood dataset, providing insights into algorithm performance and feature engineering effectiveness. 