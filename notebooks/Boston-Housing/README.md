# Boston Housing Regression Project

This folder contains a comprehensive machine learning analysis of the Boston Housing dataset using our custom ML algorithms implemented from scratch.

## Dataset

The **Boston Housing dataset** is a classic machine learning dataset containing housing information for 506 census tracts in Boston suburbs:
- **Target Variable**: MEDV (Median home value in $1000s)
- **Features**: 13 housing-related features
- **Problem Type**: Regression (predicting house prices)

### Features
- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxides concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to employment centres
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property-tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)² where Bk is the proportion of blacks by town
- **LSTAT**: % lower status of the population

## Project Structure

```
Boston-Housing/
├── 01-Boston-Housing-EDA-Cleaning.ipynb    # Data exploration and preprocessing
├── 02-Boston-Housing-Modelling.ipynb       # ML algorithm applications
└── README.md                               # This file
```

## Data Processing Pipeline

### 1. Exploratory Data Analysis (`01-Boston-Housing-EDA-Cleaning.ipynb`)
- **Data Loading**: Load Boston Housing dataset from CSV
- **Missing Values**: Check and handle missing data (20 missing values in 6 features)
- **Data Distribution**: Analyze feature distributions and target variable
- **Correlation Analysis**: Examine feature relationships with target
- **Outlier Detection**: Identify and analyze outliers using IQR method
- **Feature Engineering**: Create new features:
  - Crime rate categories
  - Room size categories
  - Age categories
  - Distance categories
  - Tax rate categories
  - Socioeconomic status categories
- **Data Scaling**: Apply StandardScaler for normalization
- **Train-Test Split**: 80-20 split
- **Data Export**: Save processed data to `../../data/HousingProcessed/`

### 2. Machine Learning Modeling (`02-Boston-Housing-Modelling.ipynb`)

#### Regression Algorithms
1. **Linear Regression** (Gradient descent optimization)
2. **Neural Network** (Multi-layer with ReLU activation)
3. **Random Forest** (100 trees with feature importance)
4. **Decision Tree** (Regression tree)
5. **Support Vector Machine** (Regression with RBF kernel)
6. **XGBoost** (Gradient boosting)
7. **K-Nearest Neighbors** (Regression with k=5)

#### Unsupervised Learning Algorithms
1. **K-Means Clustering** (for feature analysis)
2. **Principal Component Analysis** (Dimensionality reduction)

## Key Features

### Feature Engineering
The project includes comprehensive feature engineering:
- **Categorical features** created from continuous variables
- **Binning strategies** for crime rates, room sizes, ages
- **Distance-based features** for accessibility
- **Socioeconomic indicators** for neighborhood analysis

### Model Evaluation
- **R-squared (R²) Score** for regression performance
- **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)** for robust evaluation
- **Feature importance analysis** (Random Forest, XGBoost)
- **Model comparison** and ranking
- **Residual analysis** for best performing model

### Visualization
- **Feature distributions** and histograms
- **Correlation heatmaps** and scatter plots
- **Training history plots** (Neural Network)
- **Feature importance plots** (Random Forest, XGBoost)
- **Residual plots** and Q-Q plots
- **Prediction vs Actual plots**

## Expected Results

The Boston Housing dataset typically shows:
- **Moderate to high R² scores** (0.6-0.9 range)
- **Clear feature importance** patterns (LSTAT, RM, NOX often top)
- **Some non-linear relationships** requiring advanced algorithms
- **Outlier sensitivity** in certain models

## Usage

1. **Run EDA notebook first**: `01-Boston-Housing-EDA-Cleaning.ipynb`
2. **Run modeling notebook**: `02-Boston-Housing-Modelling.ipynb`
3. **Analyze results** and compare model performances

## Data Files

- **Raw data**: `../../data/HousingData.csv`
- **Processed data**: `../../data/HousingProcessed/`
  - `X_train.csv`, `X_test.csv`
  - `y_train.csv`, `y_test.csv`
  - `preprocessing_info.pkl`

## Technical Details

- **Original features**: 13
- **Engineered features**: Additional categorical features
- **Training samples**: ~405 (80%)
- **Test samples**: ~101 (20%)
- **Target**: Continuous (house prices in $1000s)
- **Scaling**: StandardScaler applied
- **Cross-validation**: Train-test split

## Model Performance Insights

### Expected Rankings (Typical)
1. **XGBoost** - Often best performance due to non-linear relationships
2. **Random Forest** - Good performance with feature importance
3. **Neural Network** - Can capture complex patterns
4. **Support Vector Machine** - Good for non-linear regression
5. **Linear Regression** - Baseline performance
6. **Decision Tree** - Interpretable but may overfit
7. **K-Nearest Neighbors** - Simple but effective

### Key Insights
- **LSTAT** (% lower status) is typically the most important feature
- **RM** (average rooms) shows strong positive correlation with price
- **NOX** (air pollution) shows negative correlation with price
- **Non-linear algorithms** often outperform linear regression
- **Feature engineering** improves model performance

## Challenges and Considerations

### Data Quality
- **Missing values** in 6 features (handled with mean imputation)
- **Outliers** in crime rates and property values
- **Feature scaling** required due to different scales

### Model Selection
- **Non-linear relationships** favor tree-based and neural network models
- **Feature importance** helps with interpretability
- **Overfitting** risk with complex models on small dataset

### Business Context
- **House price prediction** has real-world applications
- **Feature interpretability** important for stakeholders
- **Error analysis** helps understand prediction accuracy

This project demonstrates the application of all custom ML algorithms to a classic regression problem, providing insights into algorithm performance, feature engineering effectiveness, and real-world housing price prediction. 