#!/usr/bin/env python3
"""
Generate Iris dataset and save it to CSV format
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

def generate_iris_data():
    """Generate and save Iris dataset"""
    
    # Load Iris dataset
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = iris.target_names[iris.target]
    
    # Save to CSV
    output_file = "../data/IrisData.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Iris dataset saved to: {output_file}")
    print(f"Shape: {df.shape}")
    print(f"Features: {list(iris.feature_names)}")
    print(f"Target: species")
    print(f"Classes: {list(iris.target_names)}")
    
    # Display sample
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Display statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    return df

if __name__ == "__main__":
    generate_iris_data() 