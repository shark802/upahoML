#!/usr/bin/env python3
"""
Flask API Server for Land Cost Prediction
Deployed on Heroku
"""

import os
import json
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from land_predictions import LandPredictions
from ml_predictive_analytics import MLPredictiveAnalytics

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize models (lazy loading)
land_predictions = None
ml_analytics = None

def get_db_config():
    """Get database configuration from environment or config file"""
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
            'database': os.environ.get('DB_NAME', 'u520834156_dbUPAHOZoning')
        }

def init_land_predictions():
    """Initialize land predictions model"""
    global land_predictions
    if land_predictions is None:
        try:
            db_config = get_db_config()
            land_predictions = LandPredictions(db_config)
            if not land_predictions.load_models():
                app.logger.warning("Land prediction models not loaded. Train models first.")
        except Exception as e:
            app.logger.error(f"Error initializing land predictions: {e}")
            raise
    return land_predictions

def init_ml_analytics():
    """Initialize ML analytics model"""
    global ml_analytics
    if ml_analytics is None:
        try:
            db_config = get_db_config()
            ml_analytics = MLPredictiveAnalytics(db_config)
            if not ml_analytics.load_models():
                app.logger.warning("ML analytics models not loaded. Train models first.")
        except Exception as e:
            app.logger.error(f"Error initializing ML analytics: {e}")
            raise
    return ml_analytics

@app.route('/')
def index():
    """Serve the land cost prediction UI"""
    return send_from_directory(os.path.dirname(__file__), 'land_cost_prediction_ui.html')

@app.route('/api')
def api_info():
    """API information endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'UPAHO Land Cost Prediction API',
        'endpoints': {
            '/predict/land_cost': 'POST - Predict current land cost',
            '/predict/land_cost_future': 'POST - Predict future land cost (5-10 years)',
            '/predict/approval': 'POST - Predict application approval probability',
            '/predict/processing_time': 'POST - Predict processing time'
        }
    })

@app.route('/predict/land_cost', methods=['POST'])
def predict_land_cost():
    """Predict current land cost"""
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'success': False, 'error': 'Invalid request data'}), 400
        
        land_data = data.get('data', {})
        lp = init_land_predictions()
        
        result = lp.predict_land_cost(land_data)
        if result:
            return jsonify({'success': True, 'prediction': result})
        else:
            return jsonify({'success': False, 'error': 'Prediction failed'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/land_cost_future', methods=['POST'])
def predict_land_cost_future():
    """Predict future land cost (5-10 years)"""
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'success': False, 'error': 'Invalid request data'}), 400
        
        land_data = data.get('data', {})
        target_years = data.get('target_years', 10)
        
        lp = init_land_predictions()
        result = lp.predict_land_cost_future(land_data, target_years)
        
        if result:
            return jsonify({'success': True, 'prediction': result})
        else:
            return jsonify({'success': False, 'error': 'Future prediction failed'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/approval', methods=['POST'])
def predict_approval():
    """Predict application approval probability"""
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'success': False, 'error': 'Invalid request data'}), 400
        
        application_data = data.get('data', {})
        ml = init_ml_analytics()
        
        result = ml.predict_approval_probability(application_data)
        if result:
            return jsonify({'success': True, 'prediction': result})
        else:
            return jsonify({'success': False, 'error': 'Prediction failed'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/processing_time', methods=['POST'])
def predict_processing_time():
    """Predict processing time"""
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'success': False, 'error': 'Invalid request data'}), 400
        
        application_data = data.get('data', {})
        ml = init_ml_analytics()
        
        result = ml.predict_processing_time(application_data)
        if result:
            return jsonify({'success': True, 'prediction': result})
        else:
            return jsonify({'success': False, 'error': 'Prediction failed'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

