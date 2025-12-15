#!/usr/bin/env python3
"""
Model Comparison Script for Land Cost Prediction
Compares Random Forest, Gradient Boosting, and Ensemble models with cross-validation
"""

import sys
import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from land_predictions import LandPredictions

def get_db_config():
    """Get database configuration"""
    config_file = os.path.join(os.path.dirname(__file__), 'config.json')
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            if 'port' not in config:
                config['port'] = 3306
            return config
    else:
        config = {
            'host': os.environ.get('DB_HOST', 'localhost'),
            'user': os.environ.get('DB_USER', 'root'),
            'password': os.environ.get('DB_PASSWORD', ''),
            'database': os.environ.get('DB_NAME', 'u520834156_dbUPAHOZoning'),
            'port': int(os.environ.get('DB_PORT', 3306))
        }
        return config

def compare_models(cv_folds=5):
    """
    Compare different models for land cost prediction
    
    Args:
        cv_folds: Number of folds for cross-validation (default: 5)
    
    Returns:
        Dictionary with comparison results
    """
    print("=" * 80)
    print("MODEL COMPARISON FOR LAND COST PREDICTION")
    print("=" * 80)
    print(f"Cross-Validation: {cv_folds}-fold")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Initialize LandPredictions
    db_config = get_db_config()
    lp = LandPredictions(db_config)
    
    # Load and preprocess data
    print("\n[1/4] Loading data...")
    df = lp.load_land_data()
    
    if df is None or len(df) == 0:
        return {"error": "No land cost data available"}
    
    print(f"   Loaded {len(df)} records")
    
    print("\n[2/4] Preprocessing data...")
    X, y, features = lp.preprocess_land_cost_data(df)
    
    if X is None or len(X) == 0:
        return {"error": "No valid data after preprocessing"}
    
    print(f"   Training samples: {len(X)}")
    print(f"   Features: {len(features)}")
    
    # Scale features (needed for all models)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Models to compare - create model instances
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
    from sklearn.linear_model import LinearRegression
    
    models_to_test = {
        'random_forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ),
        'ensemble': VotingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1)),
                ('gb', GradientBoostingRegressor(n_estimators=150, max_depth=8, learning_rate=0.1, subsample=0.8, random_state=42))
            ],
            weights=[0.5, 0.5]
        ),
        'linear': LinearRegression()
    }
    
    # Metrics to calculate
    scoring = {
        'r2': 'r2',
        'neg_rmse': make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred))),
        'neg_mae': make_scorer(lambda y_true, y_pred: -mean_absolute_error(y_true, y_pred))
    }
    
    results = {}
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    print("\n[3/4] Training and evaluating models...")
    print("-" * 80)
    
    for model_name, model in models_to_test.items():
        print(f"\nEvaluating {model_name.upper()}...")
        start_time = time.time()
        
        model_results = {}
        
        # Cross-validation for each metric
        for metric_name, scorer in scoring.items():
            cv_scores = cross_val_score(
                model, X_scaled, y, 
                cv=kfold, 
                scoring=scorer, 
                n_jobs=-1
            )
            
            if metric_name == 'r2':
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                model_results[metric_name] = {
                    'mean': float(mean_score),
                    'std': float(std_score),
                    'scores': [float(s) for s in cv_scores]
                }
            else:  # neg_rmse, neg_mae (negative because higher is better)
                mean_score = -cv_scores.mean()  # Convert back to positive
                std_score = cv_scores.std()
                metric_display = metric_name.replace('neg_', '')
                model_results[metric_display] = {
                    'mean': float(mean_score),
                    'std': float(std_score),
                    'scores': [float(-s) for s in cv_scores]  # Convert back to positive
                }
        
        training_time = time.time() - start_time
        model_results['training_time'] = float(training_time)
        
        # Print results
        print(f"  R² Score: {model_results['r2']['mean']:.4f} (+/- {model_results['r2']['std']*2:.4f})")
        print(f"  RMSE: {model_results['rmse']['mean']:.2f} (+/- {model_results['rmse']['std']*2:.2f}) PHP/sqm")
        print(f"  MAE: {model_results['mae']['mean']:.2f} (+/- {model_results['mae']['std']*2:.2f}) PHP/sqm")
        print(f"  Training Time: {training_time:.2f} seconds")
        
        results[model_name] = model_results
    
    print("\n" + "=" * 80)
    print("[4/4] COMPARISON SUMMARY")
    print("=" * 80)
    
    # Find best model for each metric
    best_r2 = max(results.items(), key=lambda x: x[1]['r2']['mean'])
    best_rmse = min(results.items(), key=lambda x: x[1]['rmse']['mean'])
    best_mae = min(results.items(), key=lambda x: x[1]['mae']['mean'])
    
    print(f"\nBest R² Score: {best_r2[0].upper()} ({best_r2[1]['r2']['mean']:.4f})")
    print(f"Best RMSE: {best_rmse[0].upper()} ({best_rmse[1]['rmse']['mean']:.2f} PHP/sqm)")
    print(f"Best MAE: {best_mae[0].upper()} ({best_mae[1]['mae']['mean']:.2f} PHP/sqm)")
    
    # Create summary table
    print("\n" + "-" * 80)
    print(f"{'Model':<20} {'R² Score':<15} {'RMSE':<15} {'MAE':<15} {'Time (s)':<10}")
    print("-" * 80)
    for model_name, model_results in results.items():
        r2 = model_results['r2']['mean']
        rmse = model_results['rmse']['mean']
        mae = model_results['mae']['mean']
        time_taken = model_results['training_time']
        print(f"{model_name:<20} {r2:<15.4f} {rmse:<15.2f} {mae:<15.2f} {time_taken:<10.2f}")
    
    # Add summary to results
    results['summary'] = {
        'best_r2_model': best_r2[0],
        'best_r2_score': float(best_r2[1]['r2']['mean']),
        'best_rmse_model': best_rmse[0],
        'best_rmse_score': float(best_rmse[1]['rmse']['mean']),
        'best_mae_model': best_mae[0],
        'best_mae_score': float(best_mae[1]['mae']['mean']),
        'cv_folds': cv_folds,
        'total_samples': len(X),
        'features_count': len(features),
        'comparison_date': datetime.now().isoformat()
    }
    
    # Save results to file
    output_file = 'model_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 80)
    
    return results

if __name__ == '__main__':
    try:
        results = compare_models(cv_folds=5)
        if 'error' in results:
            print(f"\nError: {results['error']}")
            sys.exit(1)
        else:
            print("\nModel comparison completed successfully!")
            sys.exit(0)
    except Exception as e:
        print(f"\nError during model comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

