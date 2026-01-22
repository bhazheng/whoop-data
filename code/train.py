#!/usr/bin/env python3
"""
Production-grade training script using Scikit-Learn Pipelines.
Implements StandardScaler for numericals and OneHotEncoder for categoricals.
Supports Multiclass Classification (3 classes) and saves SEPARATE model files.
"""

import argparse
import logging
import os
import pickle
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    df = pd.read_csv(file_path)
    
    # Target Generation: Recovery Status (3 Classes)
    # 0: Rest/Light (Recovery < 33%)
    # 1: Moderate (33% <= Recovery < 67%)
    # 2: Push Hard (Recovery >= 67%)
    if 'recovery_score' in df.columns:
        conditions = [
            (df['recovery_score'] < 33),
            (df['recovery_score'] >= 33) & (df['recovery_score'] < 67),
            (df['recovery_score'] >= 67)
        ]
        choices = [0, 1, 2] # 0: Red, 1: Yellow, 2: Green
        df['target'] = np.select(conditions, choices, default=1)
        df = df.drop(columns=['recovery_score'])
    
    if 'weight_kg' in df.columns and 'height_cm' in df.columns:
        df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, list, list]:
    # Filtering
    if 'sleep_hours' in df.columns:
        df = df[df['sleep_hours'] <= 12]
    if 'avg_heart_rate' in df.columns:
        df = df[df['avg_heart_rate'] <= 220]
        
    # Outlier Removal
    for col in ['hrv', 'resting_heart_rate', 'day_strain', 'calories_burned']:
        if col in df.columns:
             lower = df[col].quantile(0.01)
             upper = df[col].quantile(0.99)
             df = df[(df[col] >= lower) & (df[col] <= upper)]
        
    # Lowercase
    if 'gender' in df.columns:
        df['gender'] = df['gender'].str.lower()
    if 'fitness_level' in df.columns:
        df['fitness_level'] = df['fitness_level'].str.lower()

    # 1. Feature Engineering
    if 'weight_kg' in df.columns and 'height_cm' in df.columns:
        df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
        
    if 'age' in df.columns:
        try:
            bins = [17, 25, 35, 45, 55, 100]
            labels = ['18-25', '26-35', '36-45', '46-55', '56+']
            df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels).astype(str)
        except:
             df['age_group'] = 'Unknown'
            
    if 'day_strain' in df.columns and 'activity_strain' in df.columns:
        df['training_load'] = df['day_strain'] + df['activity_strain']
        
    cols_sleep = ['sleep_efficiency', 'deep_sleep_hours', 'rem_sleep_hours', 'wake_ups']
    if all(c in df.columns for c in cols_sleep):
        df['sleep_quality_score'] = df['sleep_efficiency'] + df['deep_sleep_hours'] + df['rem_sleep_hours'] - (df['wake_ups'] * 0.2)
        
    if 'hrv' in df.columns and 'resting_heart_rate' in df.columns:
        df['hrv_rhr_ratio'] = df['hrv'] / df['resting_heart_rate']
        
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['date'].dt.month.astype(str)
        df['day_of_week_num'] = df['date'].dt.dayofweek.astype(str)

    # 2. Imputation
    if 'workout_time_of_day' in df.columns:
        df['workout_time_of_day'] = df['workout_time_of_day'].fillna('No Workout')

    # 3. Leakage Removal
    leakage_cols_to_drop = [
        'date', 'user_id', 
        'calories_burned', 'workout_completed', 'activity_type', 'activity_duration_min',
        'avg_heart_rate', 'max_heart_rate', 'activity_calories',
        'hr_zone_1_min', 'hr_zone_2_min', 'hr_zone_3_min', 'hr_zone_4_min', 'hr_zone_5_min'
    ]
    df = df.drop(columns=[c for c in leakage_cols_to_drop if c in df.columns])
    
    # 4. Separate X, y
    if 'target' not in df.columns:
        raise ValueError("Target column missing after preprocessing")
        
    y = df['target'].values.astype(int)
    X = df.drop(columns=['target'])
    
    # 5. Define Column Groups
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    return X, y, numerical_cols, categorical_cols


def create_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor


def train_model(X, y, preprocessor, model_type='lgbm'):
    if model_type == 'lr':
        clf = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
    elif model_type == 'mlp':
        clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    else:
        # LightGBM
        clf = lgb.LGBMClassifier(n_estimators=100, objective='multiclass', num_class=3, random_state=42, verbose=-1)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    pipeline.fit(X, y)
    return pipeline


def evaluate(pipeline, X, y, name="Test"):
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)
    
    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_proba, multi_class='ovr')
    except:
        auc = 0.0
        
    logger.info(f"[{name}] Accuracy: {acc:.4f} | AUC (OvR): {auc:.4f}")
    return auc


def main(args):
    df = load_data(args.data_path)
    X, y, num_cols, cat_cols = preprocess_data(df)
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    preprocessor = create_preprocessor(num_cols, cat_cols)
    
    # Train All 3 Models
    logger.info("Training Logistic Regression...")
    lr_pipe = train_model(X_train, y_train, preprocessor, 'lr')
    evaluate(lr_pipe, X_val, y_val, "LR Val")
    
    logger.info("Training LightGBM...")
    lgbm_pipe = train_model(X_train, y_train, preprocessor, 'lgbm')
    evaluate(lgbm_pipe, X_val, y_val, "LGBM Val")
    
    logger.info("Training MLP...")
    mlp_pipe = train_model(X_train, y_train, preprocessor, 'mlp')
    evaluate(mlp_pipe, X_val, y_val, "MLP Val")
    
    # Retrain on Full Data for Export
    logger.info("Retraining all models on Train+Val for export...")
    X_full = pd.concat([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    
    columns = list(X_full.columns)
    
    # Output Directory
    out_dir = '../output/model'
    if not os.path.exists(out_dir):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(base_dir, 'output', 'model')
        os.makedirs(out_dir, exist_ok=True)

    specs = [('lr', 'lr'), ('lgbm', 'lgbm'), ('mlp', 'mlp')]
    
    for name, m_type in specs:
        # Train
        pipe = train_model(X_full, y_full, preprocessor, m_type)
        evaluate(pipe, X_test, y_test, f"Final Test {name}")
        
        # Save Individually
        out_path = os.path.join(out_dir, f'model_{name}.bin')
        with open(out_path, 'wb') as f:
            # We save the pipeline AND the expected columns for that pipeline (which are static here)
            pickle.dump({'pipeline': pipe, 'columns': columns}, f)
        logger.info(f"Saved {name} model to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_path = os.path.join(base, 'data', 'whoop_fitness_dataset_100k.csv')
    parser.add_argument('--data_path', default=default_path)
    main(parser.parse_args())
