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

def init_land_predictions(auto_train=True):
    """Initialize land predictions model with auto-training if needed"""
    global land_predictions
    if land_predictions is None:
        try:
            db_config = get_db_config()
            
            # Test database connection first
            try:
                import mysql.connector
                test_conn = mysql.connector.connect(
                    host=db_config.get('host', 'localhost'),
                    user=db_config.get('user', 'root'),
                    password=db_config.get('password', ''),
                    database=db_config.get('database', 'u520834156_dbUPAHOZoning'),
                    connect_timeout=5
                )
                test_conn.close()
                app.logger.info("Database connection successful")
            except Exception as db_error:
                app.logger.error(f"Database connection failed: {db_error}")
                raise Exception(f"Database connection error: {db_error}")
            
            land_predictions = LandPredictions(db_config)
            
            # Try to load models
            models_loaded = land_predictions.load_models(verbose=True)
            
            if not models_loaded and auto_train:
                app.logger.warning("No trained models found. Attempting to train models...")
                try:
                    # Try to train models from database
                    training_result = land_predictions.train_all_models()
                    if 'error' not in training_result.get('land_cost', {}):
                        app.logger.info("Models trained successfully")
                        # Reload models after training
                        land_predictions.load_models(verbose=True)
                    else:
                        app.logger.error(f"Model training failed: {training_result}")
                except Exception as train_error:
                    app.logger.error(f"Error during auto-training: {train_error}")
                    app.logger.warning("Continuing without trained models. Use /api/train endpoint to train manually.")
            elif not models_loaded:
                app.logger.warning("Land prediction models not loaded. Use /api/train endpoint to train models.")
                
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
    """Predict current land cost - Compatible with PHP API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid JSON input'}), 400
        
        # Support both direct API calls and PHP-style calls
        prediction_type = data.get('prediction_type', 'land_cost')
        land_data = data.get('data', {})
        
        # If data is at root level (direct API call), use it
        if not land_data and 'lot_area' in data:
            land_data = data
        
        lp = init_land_predictions(auto_train=True)
        
        if not lp.models.get('land_cost'):
            return jsonify({
                'success': False, 
                'error': 'No trained models available. Please train models first using /api/train endpoint or ensure database has sufficient data.'
            }), 500
        
        result = lp.predict_land_cost(land_data)
        if result:
            return jsonify({'success': True, 'prediction': result})
        else:
            return jsonify({'success': False, 'error': 'Prediction failed'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/land_cost_future', methods=['POST'])
def predict_land_cost_future():
    """Predict future land cost (5-10 years) - Compatible with PHP API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid JSON input'}), 400
        
        # Support both direct API calls and PHP-style calls
        prediction_type = data.get('prediction_type', 'land_cost_future')
        land_data = data.get('data', {})
        target_years = data.get('target_years', 10)
        
        # If data is at root level (direct API call), use it
        if not land_data and 'lot_area' in data:
            land_data = data
            target_years = data.get('target_years', 10)
        
        lp = init_land_predictions(auto_train=True)
        
        if not lp.models.get('land_cost'):
            return jsonify({
                'success': False, 
                'error': 'No trained models available. Please train models first using /api/train endpoint or ensure database has sufficient data.'
            }), 500
        
        result = lp.predict_land_cost_future(land_data, target_years)
        
        if result:
            return jsonify({'success': True, 'prediction': result})
        else:
            return jsonify({'success': False, 'error': 'Future prediction failed'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Universal prediction endpoint - Compatible with PHP land_cost_predict.py format
    Accepts: {'prediction_type': 'land_cost' | 'land_cost_future', 'data': {...}, 'target_years': 10}
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid JSON input'}), 400
        
        prediction_type = data.get('prediction_type')
        land_data = data.get('data', {})
        target_years = data.get('target_years', 10)
        
        if not prediction_type:
            return jsonify({'success': False, 'error': 'prediction_type is required'}), 400
        
        lp = init_land_predictions(auto_train=True)
        
        if not lp.models.get('land_cost'):
            return jsonify({
                'success': False, 
                'error': 'No trained models available. Please train models first using /api/train endpoint or ensure database has sufficient data.'
            }), 500
        
        if prediction_type == 'land_cost':
            result_obj = lp.predict_land_cost(land_data)
            if result_obj:
                return jsonify({'success': True, 'prediction': result_obj})
            else:
                return jsonify({'success': False, 'error': 'Prediction failed'}), 500
                
        elif prediction_type == 'land_cost_future':
            result_obj = lp.predict_land_cost_future(land_data, target_years)
            if result_obj:
                return jsonify({'success': True, 'prediction': result_obj})
            else:
                return jsonify({'success': False, 'error': 'Future prediction failed'}), 500
        else:
            return jsonify({'success': False, 'error': 'Invalid prediction type'}), 400
            
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

@app.route('/api/train', methods=['POST', 'GET'])
def train_models():
    """Train land cost prediction models from database"""
    try:
        app.logger.info("Training request received")
        
        # Check database connection
        db_config = get_db_config()
        try:
            import mysql.connector
            test_conn = mysql.connector.connect(
                host=db_config.get('host', 'localhost'),
                user=db_config.get('user', 'root'),
                password=db_config.get('password', ''),
                database=db_config.get('database', 'u520834156_dbUPAHOZoning'),
                connect_timeout=5
            )
            test_conn.close()
            app.logger.info("Database connection verified")
        except Exception as db_error:
            return jsonify({
                'success': False, 
                'error': f'Database connection failed: {str(db_error)}'
            }), 500
        
        # Initialize and train
        lp = LandPredictions(db_config)
        
        app.logger.info("Starting model training...")
        results = lp.train_all_models()
        
        if 'error' in results:
            return jsonify({
                'success': False,
                'error': results['error']
            }), 500
        
        # Reload models after training
        lp.load_models(verbose=True)
        
        return jsonify({
            'success': True,
            'message': 'Models trained successfully',
            'results': results
        })
        
    except Exception as e:
        app.logger.error(f"Training error: {e}")
        return jsonify({
            'success': False,
            'error': f'Training failed: {str(e)}'
        }), 500

@app.route('/api/check', methods=['GET'])
def check_status():
    """Check database connection and model status"""
    try:
        db_config = get_db_config()
        
        # Check database connection
        db_status = {'connected': False, 'error': None}
        try:
            import mysql.connector
            conn = mysql.connector.connect(
                host=db_config.get('host', 'localhost'),
                user=db_config.get('user', 'root'),
                password=db_config.get('password', ''),
                database=db_config.get('database', 'u520834156_dbUPAHOZoning'),
                connect_timeout=5
            )
            
            # Test query
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM application_forms LIMIT 1")
            count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            db_status = {
                'connected': True,
                'records_available': count,
                'host': db_config.get('host'),
                'database': db_config.get('database')
            }
        except Exception as e:
            db_status = {
                'connected': False,
                'error': str(e)
            }
        
        # Check models
        models_status = {'loaded': False, 'models_dir': None, 'files': []}
        try:
            lp = LandPredictions(db_config)
            models_dir = lp.models_dir
            models_status['models_dir'] = models_dir
            
            import os
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith(('.pkl', '.json'))]
                models_status['files'] = model_files
                models_status['loaded'] = len(model_files) > 0
                
                # Try to load
                if lp.load_models():
                    models_status['loaded'] = True
                    models_status['model_count'] = len(lp.models)
        except Exception as e:
            models_status['error'] = str(e)
        
        return jsonify({
            'success': True,
            'database': db_status,
            'models': models_status
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

