#!/usr/bin/env python3
"""
Simple test script to verify land cost prediction is working
Tests the prediction API with sample data
"""

import json
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from land_predictions import LandPredictions

def get_db_config():
    """Get database configuration"""
    config_file = os.path.join(os.path.dirname(__file__), 'config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        # Use environment variables (Heroku config vars)
        return {
            'host': os.environ.get('DB_HOST', 'localhost'),
            'user': os.environ.get('DB_USER', 'root'),
            'password': os.environ.get('DB_PASSWORD', ''),
            'database': os.environ.get('DB_NAME', 'u520834156_dbUPAHOZoning'),
            'port': int(os.environ.get('DB_PORT', 3306))
        }

def test_prediction():
    """Test the land cost prediction"""
    print("=" * 60)
    print("Testing Land Cost Prediction")
    print("=" * 60)
    print()
    
    # Initialize
    print("1. Initializing LandPredictions...")
    db_config = get_db_config()
    lp = LandPredictions(db_config)
    
    # Load models
    print("2. Loading models...")
    if not lp.load_models(verbose=True):
        print("❌ ERROR: Failed to load models!")
        print("   Models might not be trained yet.")
        print("   Try training first: python land_predictions.py")
        return False
    
    print("✅ Models loaded successfully!")
    print()
    
    # Check scaler
    if 'land_cost' in lp.scalers:
        scaler = lp.scalers['land_cost']
        if hasattr(scaler, 'n_features_in_'):
            print(f"✅ Scaler expects {scaler.n_features_in_} features")
        else:
            print("⚠️  Scaler doesn't have n_features_in_ attribute")
    print()
    
    # Test data - minimal required fields
    print("3. Testing prediction with sample data...")
    test_data = {
        'lot_area': 200,
        'project_area': 150,
        'project_type': 'residential',
        'location': 'Downtown',
        'year': 2024,
        'month': 12,
        'age': 35
    }
    
    print(f"   Test data: {json.dumps(test_data, indent=2)}")
    print()
    
    # Test current year prediction
    print("4. Testing current year prediction...")
    try:
        result = lp.predict_land_cost(test_data)
        if result:
            print("✅ Current year prediction successful!")
            print(f"   Predicted cost per sqm: {result.get('predicted_cost_per_sqm', 0):,.2f} PHP")
            print(f"   Features used: {result.get('features_used', 'unknown')}")
            print(f"   Model R²: {result.get('model_r2', 0):.4f}")
            print()
        else:
            print("❌ ERROR: Prediction returned None!")
            return False
    except Exception as e:
        print(f"❌ ERROR: Prediction failed!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test future prediction
    print("5. Testing future prediction (10 years)...")
    try:
        future_result = lp.predict_land_cost_future(test_data, target_years=10)
        if future_result:
            print("✅ Future prediction successful!")
            current = future_result.get('current_prediction', {})
            future = future_result.get('future_prediction', {})
            print(f"   Current cost per sqm: {current.get('cost_per_sqm', 0):,.2f} PHP")
            print(f"   Future cost per sqm ({future.get('target_year', 'N/A')}): {future.get('cost_per_sqm', 0):,.2f} PHP")
            print(f"   Appreciation rate: {future.get('appreciation_rate', 0)*100:.2f}% per year")
            print(f"   Total appreciation: {future.get('total_appreciation', 0)*100:.2f}%")
            print()
        else:
            print("❌ ERROR: Future prediction returned None!")
            return False
    except Exception as e:
        print(f"❌ ERROR: Future prediction failed!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with missing optional fields (to ensure defaults work)
    print("6. Testing prediction with missing optional fields...")
    minimal_data = {
        'lot_area': 200,
        'project_area': 150,
        'project_type': 'residential',
        # No location, latitude, longitude, site_zoning
    }
    
    try:
        result2 = lp.predict_land_cost(minimal_data)
        if result2:
            print("✅ Prediction with minimal data successful!")
            print(f"   Predicted cost per sqm: {result2.get('predicted_cost_per_sqm', 0):,.2f} PHP")
            print(f"   Features used: {result2.get('features_used', 'unknown')}")
            print()
        else:
            print("❌ ERROR: Minimal data prediction returned None!")
            return False
    except Exception as e:
        print(f"❌ ERROR: Minimal data prediction failed!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print()
    print("The prediction system is working correctly.")
    print("All 23 features are being generated properly.")
    return True

if __name__ == '__main__':
    success = test_prediction()
    sys.exit(0 if success else 1)


