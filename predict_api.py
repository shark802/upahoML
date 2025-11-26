#!/usr/bin/env python3
"""
Prediction API for Real-time ML Predictions
Called from PHP to get predictions for specific applications
"""

import sys
import json
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_predictive_analytics import MLPredictiveAnalytics

def main():
    """Main function to handle prediction requests"""
    if len(sys.argv) < 2:
        print(json.dumps({'success': False, 'error': 'No data file provided'}))
        sys.exit(1)
    
    data_file = sys.argv[1]
    
    try:
        # Load request data
        with open(data_file, 'r') as f:
            request_data = json.load(f)
        
        prediction_type = request_data.get('prediction_type')
        application_data = request_data.get('data', {})
        
        # Load database config
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
        
        # Initialize ML analytics
        ml_analytics = MLPredictiveAnalytics(db_config)
        
        # Load models
        if not ml_analytics.load_models():
            print(json.dumps({'success': False, 'error': 'No trained models available'}))
            sys.exit(1)
        
        # Make prediction
        if prediction_type == 'approval':
            result = ml_analytics.predict_approval_probability(application_data)
        elif prediction_type == 'processing_time':
            result = ml_analytics.predict_processing_time(application_data)
        else:
            result = {'success': False, 'error': 'Invalid prediction type'}
        
        if result:
            print(json.dumps({'success': True, 'prediction': result}))
        else:
            print(json.dumps({'success': False, 'error': 'Prediction failed'}))
            
    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}))
        sys.exit(1)

if __name__ == '__main__':
    main()

