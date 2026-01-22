#!/usr/bin/env python3
"""
Production-grade prediction service using Scikit-Learn Pipeline.
Supports Multi-Model Selection from SEPARATE files.
"""

import pickle
import logging
import os
from typing import Dict, Any, Tuple
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global state
models = {} # {'lgbm': {'pipeline': pipe, 'columns': cols}, ...}

# Class Mapping
CLASS_LABELS = {
    0: "Rest/Light",
    1: "Moderate",
    2: "Push Hard"
}

def load_models(model_dir: str = None):
    if model_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base, 'output', 'model')
    
    logger.info(f"Loading models from directory: {model_dir}")
    if not os.path.exists(model_dir):
        logger.warning(f"Model directory not found: {model_dir}")
        return

    # Pattern: model_*.bin
    files = glob.glob(os.path.join(model_dir, "model_*.bin"))
    if not files:
        logger.warning("No model_*.bin files found.")
    
    global models
    models = {}
    
    for file_path in files:
        # Extract name: model_lgbm.bin -> lgbm
        basename = os.path.basename(file_path)
        name_part = basename.replace("model_", "").replace(".bin", "")
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict) and 'pipeline' in data:
                models[name_part] = {
                    'pipeline': data['pipeline'],
                    'columns': data['columns']
                }
                logger.info(f"Loaded model: {name_part}")
            else:
                logger.error(f"Invalid format for {basename}")
                
        except Exception as e:
            logger.error(f"Failed to load {basename}: {e}")
            
    if not models:
        logger.error("No valid models loaded!")

def prepare_dataframe(data: Dict[str, Any], columns: list) -> pd.DataFrame:
    df = pd.DataFrame([data])
    
    # Feature Engineering
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
        
    cols_sleep = ['sleep_efficiency', 'deep_sleep_hours', 'rem_sleep_hours', 'wake_ups']
    if all(c in df.columns for c in cols_sleep):
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
        if col not in df.columns:
            df[col] = np.nan
            
    return df[columns]

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'available_models': list(models.keys()), 
        'service': 'Whoop AI Coach: Daily Intensity Recommender'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data: return jsonify({'error': 'No input data'}), 400
        
        # Select Model
        model_name = data.get('model', 'lgbm').lower()
        if model_name not in models:
            return jsonify({'error': f"Model '{model_name}' not found. Available: {list(models.keys())}"}), 400
        
        model_data = models[model_name]
        pipeline = model_data['pipeline']
        model_cols = model_data['columns']
        
        # Prepare
        input_data = {k: v for k, v in data.items() if k != 'model'}
        df = prepare_dataframe(input_data, model_cols)
        
        # Predict
        pred_idx = int(pipeline.predict(df)[0])
        probs = pipeline.predict_proba(df)[0]
        prob = float(probs[pred_idx])
        rec = CLASS_LABELS.get(pred_idx, "Unknown")
        
        return jsonify({
            'recommendation': rec,
            'status_code': pred_idx,
            'probability': round(prob, 4),
            'model': model_name
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        load_models()
        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        logger.fatal(f"Failed to start: {e}")
        exit(1)
