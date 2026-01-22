#!/usr/bin/env python3
"""
Standalone Prediction Script for Whoop AI Coach.
Performs full feature engineering and inference identical to the production service.
Supports Multi-Model Selection from SEPARATE files.
"""

import argparse
import json
import pickle
import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Class Mapping
CLASS_LABELS = {
    0: "Rest/Light",
    1: "Moderate",
    2: "Push Hard"
}

def load_model(model_name: str, model_dir: str = None):
    # Default path
    if model_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base, 'output', 'model')
        
    file_path = os.path.join(model_dir, f"model_{model_name}.bin")
    
    logger.info(f"Loading model from {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'pipeline' in data:
        return data['pipeline'], data['columns']
    else:
        raise ValueError("Invalid model file format")

def prepare_dataframe(data: Dict[str, Any], columns: list) -> pd.DataFrame:
    df = pd.DataFrame([data])
    
    # Feature Engineering (Identical to app/main.py)
    if 'height_cm' in df.columns and 'weight_kg' in df.columns:
         try: df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
         except: pass 
         
    if 'gender' in df.columns: df['gender'] = df['gender'].astype(str).str.lower()
    if 'fitness_level' in df.columns: df['fitness_level'] = df['fitness_level'].astype(str).str.lower()

    if 'age' in df.columns:
        try:
            bins = [17, 25, 35, 45, 55, 100]
            labels = ['18-25', '26-35', '36-45', '46-55', '56+']
            df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels).astype(str)
        except: df['age_group'] = 'Unknown'
            
    if 'day_strain' in df.columns and 'activity_strain' in df.columns:
        try: df['training_load'] = df['day_strain'] + df['activity_strain']
        except: pass
        
    if all(c in df.columns for c in ['sleep_efficiency', 'deep_sleep_hours', 'rem_sleep_hours', 'wake_ups']):
        try: df['sleep_quality_score'] = df['sleep_efficiency'] + df['deep_sleep_hours'] + df['rem_sleep_hours'] - (df['wake_ups'] * 0.2)
        except: pass
        
    if 'hrv' in df.columns and 'resting_heart_rate' in df.columns:
        try: df['hrv_rhr_ratio'] = df['hrv'] / df['resting_heart_rate']
        except: pass
        
    if 'date' in df.columns:
        try:
            dt = pd.to_datetime(df['date'], errors='coerce')
            df['month'] = dt.dt.month.astype(str)
            df['day_of_week_num'] = dt.dt.dayofweek.astype(str)
        except: pass

    if 'workout_time_of_day' in df.columns:
        df['workout_time_of_day'] = df['workout_time_of_day'].fillna('No Workout')
    elif 'workout_time_of_day' in columns:
        df['workout_time_of_day'] = 'No Workout'

    for col in columns:
        if col not in df.columns: df[col] = np.nan
            
    return df[columns]

def main(args):
    # Load Input
    data = {}
    if args.file:
        with open(args.file, 'r') as f: data = json.load(f)
    elif args.input:
        data = json.loads(args.input)
    else:
        logger.error("Must provide --file or --input")
        return

    model_name = args.model.lower()
    
    # Load Model
    try:
        pipeline, model_columns = load_model(model_name, args.model_dir)
    except Exception as e:
        logger.fatal(f"Failed to load model '{model_name}': {e}")
        return

    # Predict
    try:
        input_data = {k: v for k, v in data.items() if k != 'model'}
        df = prepare_dataframe(input_data, model_columns)
        
        pred_idx = int(pipeline.predict(df)[0])
        probs = pipeline.predict_proba(df)[0]
        prob = float(probs[pred_idx])
        rec = CLASS_LABELS.get(pred_idx, "Unknown")
        
        result = {
            'recommendation': rec,
            'status_code': pred_idx,
            'probability': round(prob, 4),
            'model': model_name,
            'service': 'Whoop AI Coach: Daily Intensity Recommender'
        }
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.fatal(f"Prediction failed: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Whoop AI Coach Prediction Tool")
    parser.add_argument('--input', type=str, help='JSON string input')
    parser.add_argument('--file', type=str, help='Path to JSON input file')
    parser.add_argument('--model', type=str, default='lgbm', help='Model to use (lgbm, lr, mlp)')
    parser.add_argument('--model_dir', type=str, help='Directory containing model files (optional)')
    
    main(parser.parse_args())
