#!/usr/bin/env python3
"""
Land Cost Prediction API
Called from PHP to predict land cost
"""

import sys
import json
import os
import io
import contextlib

# Suppress all print statements and redirect stderr
# This ensures only JSON is output
class SuppressOutput:
    def __init__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
    
    def __enter__(self):
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self
    
    def __exit__(self, *args):
        sys.stdout = self.stdout
        sys.stderr = self.stderr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from land_predictions import LandPredictions

def main():
    if len(sys.argv) < 2:
        print(json.dumps({'success': False, 'error': 'No data file provided'}))
        sys.exit(1)
    
    data_file = sys.argv[1]
    
    try:
        with open(data_file, 'r') as f:
            request_data = json.load(f)
        
        prediction_type = request_data.get('prediction_type')
        land_data = request_data.get('data', {})
        
        config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                db_config = json.load(f)
        else:
            db_config = {
                'host': 'localhost',
                'user': 'root',
                'password': '',
                'database': 'u520834156_dbUPAHOZoning'
            }
        
        # Suppress all output during model loading and prediction
        with SuppressOutput():
            land_predictions = LandPredictions(db_config)
            
            if not land_predictions.load_models():
                result = {'success': False, 'error': 'No trained models available'}
            elif prediction_type == 'land_cost':
                result_obj = land_predictions.predict_land_cost(land_data)
                if result_obj:
                    result = {'success': True, 'prediction': result_obj}
                else:
                    result = {'success': False, 'error': 'Prediction failed'}
            elif prediction_type == 'land_cost_future':
                # Long-term prediction (5-10 years)
                target_years = request_data.get('target_years', 10)
                result_obj = land_predictions.predict_land_cost_future(land_data, target_years)
                if result_obj:
                    result = {'success': True, 'prediction': result_obj}
                else:
                    result = {'success': False, 'error': 'Future prediction failed'}
            else:
                result = {'success': False, 'error': 'Invalid prediction type'}
        
        # Output only JSON (no debug messages)
        print(json.dumps(result))
            
    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}))
        sys.exit(1)

if __name__ == '__main__':
    main()

