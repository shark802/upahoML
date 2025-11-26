#!/usr/bin/env python3
"""
Time Series Forecasting API
Called from PHP to get time series forecasts
"""

import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_predictive_analytics import MLPredictiveAnalytics

def main():
    """Main function for time series forecasting"""
    if len(sys.argv) < 2:
        print(json.dumps({'success': False, 'error': 'No data file provided'}))
        sys.exit(1)
    
    data_file = sys.argv[1]
    
    try:
        with open(data_file, 'r') as f:
            request_data = json.load(f)
        
        zone_type = request_data.get('zone_type', 'residential')
        historical_data = request_data.get('historical_data', [])
        forecast_months = request_data.get('forecast_months', 12)
        
        # Load database config
        config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                db_config = json.load(f)
        else:
            db_config = {
                 'host': '153.92.15.8',
            'user': 'u520834156_uPAHOZone25',
            'password': 'Y+;a+*1y',
            'database': 'u520834156_dbUPAHOZoning'
            }
        
        # Initialize ML analytics
        ml_analytics = MLPredictiveAnalytics(db_config)
        
        # Convert historical data to DataFrame
        import pandas as pd
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Forecast
        result = ml_analytics.train_time_series_model(
            df,
            target_column='applications_submitted',
            model_type='prophet',
            periods=forecast_months
        )
        
        if result:
            print(json.dumps({'success': True, 'forecast': result}))
        else:
            print(json.dumps({'success': False, 'error': 'Forecasting failed'}))
            
    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}))

if __name__ == '__main__':
    main()

