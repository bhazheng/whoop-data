# Exported from notebooks/notebook.ipynb
# run this script to execute the notebook logic without Jupyter


# [Markdown]
# # Whoop AI Coach: Daily Intensity Recommender
# ## Comprehensive EDA and Model Selection


# --- Cell 1 ---


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# [Markdown]
# ## 1. Setup and Data Loading <a name='setup'></a>


# --- Cell 4 ---


# Load dataset with robust path handling
import os
import pandas as pd

# Try multiple potential paths
possible_paths = [
    '../data/whoop_fitness_dataset_100k.csv',      # Standard Jupyter (relative to notebooks/)
    'data/whoop_fitness_dataset_100k.csv',         # VS Code / Root Context
    'whoop_fitness_dataset_100k.csv'               # Same directory
]

file_path = None
for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    # Print CWD for debugging
    print(f"Current Working Directory: {os.getcwd()}")
    raise FileNotFoundError(f"Dataset not found. Tried: {possible_paths}")

print(f"Loading data from: {file_path}")
df = pd.read_csv(file_path)
print(f"Dataset Shape: {df.shape}")
df.head()


# [Markdown]
# ## 2. Data Preparation and Cleaning <a name='prep'></a>


# --- Cell 6 ---


# Check missing values
print("Missing Values:")
print(df.isnull().sum())
print(f"\nTotal duplicates: {df.duplicated().sum()}")

# --- IMPUTATION START ---
# Impute missing values to prevent model errors
print('Imputing missing values...')
numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print('Missing values after imputation:', df.isnull().sum().sum())
# --- IMPUTATION END ---


# --- Cell 7 ---

# Analyze workout_time_of_day
print("Original workout_time_of_day counts:")
print(df['workout_time_of_day'].value_counts())

# Fill missing workout_time_of_day logically
df['workout_time_of_day'] = df['workout_time_of_day'].fillna('No Workout')

print("\nImputed workout_time_of_day counts:")
print(df['workout_time_of_day'].value_counts())


# --- Cell 8 ---

# Target Generation: Recovery Status (3 Classes)
# 0: Rest/Light (Recovery < 33%)
# 1: Moderate (33% <= Recovery < 67%)
# 2: Push Hard (Recovery >= 67%)

conditions = [
    (df['recovery_score'] < 33),
    (df['recovery_score'] >= 33) & (df['recovery_score'] < 67),
    (df['recovery_score'] >= 67)
]
choices = [0, 1, 2] # 0: Red, 1: Yellow, 2: Green
df['target'] = np.select(conditions, choices, default=1)

# Drop source to avoid leakage
df = df.drop(columns=['recovery_score'])

print(f"Target Distribution:\n{df['target'].value_counts(normalize=True)}")


# --- Cell 9 ---


# --- Feature Engineering & Preprocessing (Parity with train.py) ---

# 1. Filtering & Outlier Removal
print(f"Shape before filtering: {df.shape}")
if 'sleep_hours' in df.columns:
    df = df[df['sleep_hours'] <= 12]
if 'avg_heart_rate' in df.columns:
    df = df[df['avg_heart_rate'] <= 220]

# Outliers (HRV, RHR only for safety in notebook)
for col in ['hrv', 'resting_heart_rate', 'day_strain', 'calories_burned']:
    if col in df.columns:
         lower = df[col].quantile(0.01)
         upper = df[col].quantile(0.99)
         df = df[(df[col] >= lower) & (df[col] <= upper)]
print(f"Shape after filtering: {df.shape}")

# 2. Text Normalization
if 'gender' in df.columns:
    df['gender'] = df['gender'].str.lower()
if 'fitness_level' in df.columns:
    df['fitness_level'] = df['fitness_level'].str.lower()

# 3. BMI
if 'weight_kg' in df.columns and 'height_cm' in df.columns:
    df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

# 4. Age Group (Updated Bins from train.py)
if 'age' in df.columns:
    bins = [17, 25, 35, 45, 55, 100]
    labels = ['18-25', '26-35', '36-45', '46-55', '56+']
    try:
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels).astype(str)
    except:
        df['age_group'] = 'Unknown'

# 5. Training Load
if 'day_strain' in df.columns and 'activity_strain' in df.columns:
    df['training_load'] = df['day_strain'] + df['activity_strain']

# 6. Sleep Quality Score
cols_sleep = ['sleep_efficiency', 'deep_sleep_hours', 'rem_sleep_hours', 'wake_ups']
if all(c in df.columns for c in cols_sleep):
    df['sleep_quality_score'] = df['sleep_efficiency'] + df['deep_sleep_hours'] + df['rem_sleep_hours'] - (df['wake_ups'] * 0.2)

# 7. HRV / RHR Ratio
if 'hrv' in df.columns and 'resting_heart_rate' in df.columns:
    df['hrv_rhr_ratio'] = df['hrv'] / df['resting_heart_rate']

# 8. Date Features
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.month.astype(str)
    df['day_of_week_num'] = df['date'].dt.dayofweek.astype(str)

print("Feature Engineering Complete.")


# [Markdown]
# ## 3. Exploratory Data Analysis (EDA) <a name='eda'></a>


# [Markdown]
# ### Target Distribution <a name='target'></a>


# --- Cell 12 ---


plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df, palette='viridis')
plt.title('Target Distribution (High Recovery vs Low Recovery)', fontsize=15)
plt.xlabel('High Recovery (1) / Low Recovery (0)')
plt.ylabel('Count')
plt.show()


# [Markdown]
# ### Numerical Features <a name='num'></a>


# --- Cell 14 ---


numerical_cols = ['age', 'weight_kg', 'height_cm', 'bmi', 'day_strain', 
                  'sleep_hours', 'hrv', 'resting_heart_rate', 'calories_burned']

# Histograms
df[numerical_cols].hist(bins=30, figsize=(20, 15), edgecolor='black')
plt.suptitle('Distribution of Numerical Features', fontsize=20)
plt.tight_layout()
plt.show()


# --- Cell 15 ---


# Boxplots by Target
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.ravel()

for i, col in enumerate(numerical_cols):
    sns.boxplot(x='target', y=col, data=df, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{col} vs Target', fontsize=12)

plt.tight_layout()
plt.show()


# [Markdown]
# ### Categorical Features <a name='cat'></a>


# --- Cell 17 ---


categorical_cols = ['gender', 'fitness_level', 'primary_sport', 'day_of_week']

for col in categorical_cols:
    plt.figure(figsize=(12, 5))
    sns.countplot(x=col, hue='target', data=df, palette='coolwarm')
    plt.title(f'{col} Distribution by Target', fontsize=15)
    plt.legend(title='High Recovery', loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# [Markdown]
# ### Correlations <a name='corr'></a>


# --- Cell 19 ---


# Correlation Matrix
plt.figure(figsize=(15, 12))
corr_cols = numerical_cols + ['target']
corr_matrix = df[corr_cols].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Feature Correlation Matrix', fontsize=18)
plt.show()


# [Markdown]
# ## 4. Feature Importance <a name='importance'></a>


# --- Cell 21 ---


# Prepare Data for Modeling
# Drop non-feature columns and leakage
drop_cols = ['user_id', 'date', 'recovery_score', 'target'] 
# Also drop columns that might be leakage if they are outcomes of recovery (e.g. sleep performance might be highly correlated)
# For now, we keep physiological metrics.

X_cols = [c for c in df.columns if c not in drop_cols]
features = df[X_cols].copy()
target = df['target']

# Handle categorical encoding
cat_features = features.select_dtypes(include=['object']).columns.tolist()
num_features = features.select_dtypes(include=[np.number]).columns.tolist()

feat_dict = features.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_encoded = dv.fit_transform(feat_dict)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_encoded, target)

# Extract Importance
importances = rf.feature_importances_
feature_names = dv.get_feature_names_out()
feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False).head(20)

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feat_imp, palette='viridis')
plt.title('Top 20 Important Features (Random Forest)', fontsize=15)
plt.tight_layout()
plt.show()


# [Markdown]
# ## 5. Model Selection <a name='model'></a>


# [Markdown]
# # 4. Model Selection & Evaluation (Multiclass)


# --- Cell 24 ---

# --- 2. Fine-Tuning and Comparison ---

# Data Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, target, test_size=0.2, random_state=42)

# Define Models and Grids
models_params = {
    'Logistic Regression': {
        'model': LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42),
        'params': {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs']
        }
    },
    'LightGBM': {
        'model': lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42, verbose=-1),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 50]
        }
    },
    'MLP': {
        'model': make_pipeline(StandardScaler(with_mean=False), MLPClassifier(random_state=42, max_iter=500)),
        'params': {
            'mlpclassifier__hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'mlpclassifier__alpha': [0.0001, 0.001]
        }
    }
}

# Directory for saving models
output_dir = '../output/model'
os.makedirs(output_dir, exist_ok=True)

results = []
best_models = {}

# Tuning Loop
for name, config in models_params.items():
    print(f"Tuning {name}...")
    # Use GridSearchCV for thoroughness (or RandomizedSearchCV if larger grids)
    search = GridSearchCV(config['model'], config['params'], cv=3, scoring='roc_auc', n_jobs=-1)
    
    try:
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_models[name] = best_model
        
        # Evaluate on Test
        y_pred = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        results.append({'Model': name, 'AUC': auc, 'Best Params': str(search.best_params_)})
        print(f"  Best Params: {search.best_params_}")
        print(f"  Test AUC: {auc:.4f}")
        
        # Save Model
        filename = f"{name.replace(' ', '_').lower()}_tuned.bin"
        save_path = os.path.join(output_dir, filename)
        with open(save_path, 'wb') as f_out:
            pickle.dump(best_model, f_out)
        print(f"  Saved to {save_path}")
        
    except Exception as e:
        print(f"  Error tuning {name}: {e}")

# Comparison visualization
res_df = pd.DataFrame(results)
print("\nSummary Table:")
print(res_df[['Model', 'AUC', 'Best Params']])

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='AUC', data=res_df, palette='viridis')
plt.ylim(0.5, 1.0)
plt.title('Tuned Model Performance Comparison (ROC AUC)', fontsize=15)
plt.show()
