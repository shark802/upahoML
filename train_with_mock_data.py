#!/usr/bin/env python3
"""
Complete Training Script with Enhanced Mock Data
1. Generates realistic mock data with all feature engineering features
2. Inserts into database
3. Trains the machine learning model
"""

import sys
import os
import json
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_mock_data import get_db_config, connect_database, generate_mock_data, insert_mock_data
from land_predictions import LandPredictions

def main():
    print("=" * 70)
    print("LAND PROPERTY VALUE PREDICTION - TRAINING WITH ENHANCED MOCK DATA")
    print("=" * 70)
    
    # Step 1: Get database configuration
    print("\n[Step 1/4] Getting database configuration...")
    db_config = get_db_config()
    print(f"[OK] Database: {db_config.get('database')}")
    print(f"[OK] Host: {db_config.get('host')}")
    
    # Step 2: Connect to database
    print("\n[Step 2/4] Connecting to database...")
    connection = connect_database(db_config)
    if not connection:
        print("[ERROR] Failed to connect to database")
        return False
    print("[OK] Database connection successful")
    
    # Step 3: Generate and insert enhanced mock data
    print("\n[Step 3/4] Generating and inserting enhanced mock data...")
    print("  This data includes all feature engineering features:")
    print("    - Latitude/Longitude (for distance calculations)")
    print("    - Site Zoning (for zoning categories)")
    print("    - Location Type (for location classification)")
    print("    - All temporal features (year, month, season, etc.)")
    print("    - All size features (ratios, categories)")
    print("    - All interaction features")
    
    num_records = 2000  # Generate 2000 records for robust training
    print(f"\n  Generating {num_records} records...")
    
    data = generate_mock_data(num_records)
    print(f"[OK] Generated {len(data)} mock records")
    
    # Show data distribution
    project_type_counts = {}
    location_counts = {}
    for record in data:
        pt = record['project_type']
        loc = record['project_location']
        project_type_counts[pt] = project_type_counts.get(pt, 0) + 1
        location_counts[loc] = location_counts.get(loc, 0) + 1
    
    print("\n  Data Distribution by Project Type:")
    for pt, count in sorted(project_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {pt}: {count} records ({count/len(data)*100:.1f}%)")
    
    print("\n  Top Locations:")
    for loc, count in sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {loc}: {count} records")
    
    # Check for enhanced features
    sample_record = data[0]
    print("\n  Sample Record Features:")
    print(f"    [OK] Latitude: {sample_record.get('latitude', 'N/A')}")
    print(f"    [OK] Longitude: {sample_record.get('longitude', 'N/A')}")
    print(f"    [OK] Site Zoning: {sample_record.get('site_zoning', 'N/A')}")
    print(f"    [OK] Location Type: {sample_record.get('location_type', 'N/A')}")
    
    # Insert data
    print("\n  Inserting data into database...")
    success = insert_mock_data(connection, data, batch_size=50)
    
    if not success:
        print("[ERROR] Failed to insert mock data")
        connection.close()
        return False
    
    print("[OK] Mock data inserted successfully")
    connection.close()
    
    # Step 4: Train the machine learning model
    print("\n[Step 4/4] Training machine learning model...")
    print("  Model: Random Forest Regressor (best for realistic predictions)")
    print("  Features: 20+ engineered features")
    print("  This may take a few minutes...")
    
    try:
        # Initialize land predictions
        lp = LandPredictions(db_config)
        
        # Train all models
        print("\n  Starting model training...")
        results = lp.train_all_models()
        
        # Check results
        if 'error' in results:
            print(f"\n[ERROR] Training failed: {results['error']}")
            return False
        
        if 'land_cost' in results:
            metrics = results['land_cost']
            print("\n" + "=" * 70)
            print("TRAINING COMPLETE - MODEL PERFORMANCE")
            print("=" * 70)
            print(f"\n  Model Type: Random Forest Regressor")
            print(f"  Training Samples: {metrics.get('training_samples', 'N/A')}")
            print(f"  Test Samples: {metrics.get('test_samples', 'N/A')}")
            print(f"\n  Performance Metrics:")
            print(f"    R² Score: {metrics.get('r2_score', 0):.4f}")
            print(f"      (Higher is better, 1.0 = perfect)")
            print(f"    RMSE: {metrics.get('rmse', 0):.2f} PHP/sqm")
            print(f"      (Lower is better, average prediction error)")
            print(f"    MAE: {metrics.get('mae', 0):.2f} PHP/sqm")
            print(f"      (Lower is better, mean absolute error)")
            
            # Interpret results
            r2 = metrics.get('r2_score', 0)
            if r2 > 0.6:
                print(f"\n  [EXCELLENT] Model explains {r2*100:.1f}% of variance")
            elif r2 > 0.4:
                print(f"\n  [GOOD] Model explains {r2*100:.1f}% of variance")
            elif r2 > 0.2:
                print(f"\n  [MODERATE] Model explains {r2*100:.1f}% of variance")
                print(f"    Consider adding more training data or features")
            else:
                print(f"\n  [LOW] Model explains {r2*100:.1f}% of variance")
                print(f"    May need more data or feature engineering")
            
            # Feature importance
            if 'features' in metrics:
                print(f"\n  Features Used: {len(metrics['features'])}")
                print(f"    {', '.join(metrics['features'][:10])}...")
            
            # Save status
            if results.get('models_saved', False):
                print(f"\n  [OK] Models saved successfully")
            else:
                print(f"\n  [WARNING] Models may not have been saved")
            
            print("\n" + "=" * 70)
            print("NEXT STEPS")
            print("=" * 70)
            print("\n  1. Test predictions using the API:")
            print("     POST /predict/land_cost_future")
            print("\n  2. Check feature importance:")
            print("     Review which features matter most")
            print("\n  3. Monitor performance:")
            print("     Retrain quarterly with new data")
            print("\n  4. Add more data:")
            print("     More records = better accuracy")
            
            return True
        else:
            print("\n✗ Training completed but no results returned")
            print(f"  Results: {results}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    if success:
        print("\n[SUCCESS] All steps completed successfully!")
        sys.exit(0)
    else:
        print("\n[FAILED] Some steps failed. Check errors above.")
        sys.exit(1)

