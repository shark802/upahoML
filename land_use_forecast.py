#!/usr/bin/env python3
"""
Land Use Forecast API
Called from PHP to predict future land use distribution
"""

import sys
import json
import os

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
        
        months_ahead = request_data.get('months_ahead', 12)
        
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
        
        land_predictions = LandPredictions(db_config)
        
        predictions = land_predictions.predict_future_land_use(months_ahead)
        
        if predictions:
            print(json.dumps({
                'success': True,
                'predictions': predictions
            }))
        else:
            print(json.dumps({'success': False, 'error': 'Forecasting failed'}))
            
    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}))

if __name__ == '__main__':
    main()

