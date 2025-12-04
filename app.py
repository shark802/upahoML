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
            config = json.load(f)
            # Add port if not specified
            if 'port' not in config:
                config['port'] = 3306
            return config
    else:
        # Use environment variables (Heroku config vars)
        config = {
            'host': os.environ.get('DB_HOST', 'localhost'),
            'user': os.environ.get('DB_USER', 'root'),
            'password': os.environ.get('DB_PASSWORD', ''),
            'database': os.environ.get('DB_NAME', 'u520834156_dbUPAHOZoning'),
            'port': int(os.environ.get('DB_PORT', 3306))
        }
        return config

def init_land_predictions(auto_train=True):
    """Initialize land predictions model with auto-training if needed"""
    global land_predictions
    if land_predictions is None:
        try:
            db_config = get_db_config()
            
            # Test database connection first
            try:
                import mysql.connector
                db_host = db_config.get('host', 'localhost')
                
                # Check if using localhost (won't work on Heroku)
                if db_host == 'localhost' or db_host == '127.0.0.1':
                    raise Exception(
                        "Cannot connect to 'localhost' from Heroku. "
                        "You need a cloud database (AWS RDS, Heroku Postgres, or remote MySQL). "
                        "Set DB_HOST to your cloud database hostname. "
                        "Current DB_HOST: localhost"
                    )
                
                test_conn = mysql.connector.connect(
                    host=db_host,
                    user=db_config.get('user', 'root'),
                    password=db_config.get('password', ''),
                    database=db_config.get('database', 'u520834156_dbUPAHOZoning'),
                    connect_timeout=10,
                    port=db_config.get('port', 3306)
                )
                test_conn.close()
                app.logger.info(f"Database connection successful to {db_host}")
            except mysql.connector.Error as db_error:
                error_msg = str(db_error)
                if '2003' in error_msg or 'Can\'t connect' in error_msg:
                    raise Exception(
                        f"Database connection failed (Error 2003): Cannot connect to MySQL server at '{db_config.get('host')}'. "
                        f"Possible causes: "
                        f"1) Database host is incorrect or unreachable from Heroku, "
                        f"2) Database server is not running, "
                        f"3) Firewall is blocking Heroku IPs, "
                        f"4) Using 'localhost' (won't work on Heroku - need cloud database). "
                        f"Check your DB_HOST environment variable."
                    )
                else:
                    raise Exception(f"Database connection error: {error_msg}")
            except Exception as db_error:
                app.logger.error(f"Database connection failed: {db_error}")
                raise
            
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

@app.route('/test')
def test_page():
    """Serve the testing page for predictions"""
    return send_from_directory(os.path.dirname(__file__), 'test_predictions.html')

@app.route('/api')
def api_info():
    """API information endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'UPAHO Land Cost Prediction API',
        'endpoints': {
            '/predict/land_cost': 'POST - Predict current land cost',
            '/predict/land_cost_future': 'POST - Predict future land cost (5-10 years). Use "use_arima": true for ARIMA forecasting',
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
        use_arima = data.get('use_arima', False)  # New option to use ARIMA
        
        # If data is at root level (direct API call), use it
        if not land_data and 'lot_area' in data:
            land_data = data
            target_years = data.get('target_years', 10)
            use_arima = data.get('use_arima', False)
        
        lp = init_land_predictions(auto_train=True)
        
        # Use ARIMA if requested
        if use_arima:
            project_type = land_data.get('project_type')
            location = land_data.get('location') or land_data.get('project_location')
            forecast_months = target_years * 12
            
            result = lp.predict_land_cost_arima(
                land_data, 
                forecast_months=forecast_months,
                project_type=project_type,
                location=location
            )
            
            if result and result.get('success'):
                return jsonify({'success': True, 'prediction': result})
            elif result and not result.get('success'):
                # ARIMA failed, fallback to regular prediction
                app.logger.warning(f"ARIMA prediction failed: {result.get('error')}, falling back to regular prediction")
                use_arima = False
        
        # Regular prediction (ML-based)
        if not use_arima:
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
    Accepts: {'prediction_type': 'land_cost' | 'land_cost_future' | 'land_cost_arima', 'data': {...}, 'target_years': 10, 'use_arima': True}
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid JSON input'}), 400
        
        prediction_type = data.get('prediction_type')
        land_data = data.get('data', {})
        target_years = data.get('target_years', 10)
        use_arima = data.get('use_arima', False)
        
        if not prediction_type:
            return jsonify({'success': False, 'error': 'prediction_type is required'}), 400
        
        lp = init_land_predictions(auto_train=True)
        
        if prediction_type == 'land_cost':
            if not lp.models.get('land_cost'):
                return jsonify({
                    'success': False, 
                    'error': 'No trained models available. Please train models first using /api/train endpoint or ensure database has sufficient data.'
                }), 500
            
            result_obj = lp.predict_land_cost(land_data)
            if result_obj:
                return jsonify({'success': True, 'prediction': result_obj})
            else:
                return jsonify({'success': False, 'error': 'Prediction failed'}), 500
                
        elif prediction_type == 'land_cost_future' or prediction_type == 'land_cost_arima':
            # Use ARIMA if explicitly requested or if prediction_type is 'land_cost_arima'
            if use_arima or prediction_type == 'land_cost_arima':
                project_type = land_data.get('project_type')
                location = land_data.get('location') or land_data.get('project_location')
                forecast_months = target_years * 12
                
                result_obj = lp.predict_land_cost_arima(
                    land_data,
                    forecast_months=forecast_months,
                    project_type=project_type,
                    location=location
                )
                
                if result_obj and result_obj.get('success'):
                    return jsonify({'success': True, 'prediction': result_obj})
                elif result_obj and not result_obj.get('success'):
                    # ARIMA failed, fallback to regular prediction
                    app.logger.warning(f"ARIMA prediction failed: {result_obj.get('error')}, falling back to regular prediction")
                    use_arima = False
            
            # Regular prediction (ML-based) if ARIMA not used or failed
            if not use_arima:
                if not lp.models.get('land_cost'):
                    return jsonify({
                        'success': False, 
                        'error': 'No trained models available. Please train models first using /api/train endpoint or ensure database has sufficient data.'
                    }), 500
                
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
        
        # Check if training actually produced results
        if not results or (isinstance(results, dict) and len(results) == 0):
            # Check if there's data in database
            try:
                conn = mysql.connector.connect(
                    host=db_config.get('host'),
                    user=db_config.get('user'),
                    password=db_config.get('password'),
                    database=db_config.get('database'),
                    port=db_config.get('port', 3306),
                    connect_timeout=10
                )
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM application_forms WHERE project_cost_numeric IS NOT NULL AND lot_area IS NOT NULL AND lot_area > 0")
                count = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                
                if count == 0:
                    return jsonify({
                        'success': False,
                        'error': 'No training data available. Database has no records with cost data (project_cost_numeric and lot_area).',
                        'database_records': count
                    }), 500
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Training completed but no results returned. Check logs for details.',
                        'database_records': count,
                        'results': results
                    }), 500
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Error checking database: {str(e)}'
                }), 500
        
        if 'error' in results:
            return jsonify({
                'success': False,
                'error': results.get('error', 'Training failed'),
                'results': results
            }), 500
        
        # Reload models after training
        models_loaded = lp.load_models(verbose=True)
        
        return jsonify({
            'success': True,
            'message': 'Models trained successfully',
            'results': results,
            'models_loaded': models_loaded,
            'model_files': len(lp.models) if hasattr(lp, 'models') else 0
        })
        
    except Exception as e:
        app.logger.error(f"Training error: {e}")
        return jsonify({
            'success': False,
            'error': f'Training failed: {str(e)}'
        }), 500

@app.route('/api/insert_training_data', methods=['POST', 'GET'])
def insert_training_data():
    """Insert mock training data into database"""
    try:
        app.logger.info("Training data insertion request received")
        
        # Get number of records to generate (default 500)
        num_records = request.args.get('num_records', 500, type=int)
        if num_records > 5000:
            num_records = 5000  # Limit to prevent overload
        
        # Check database connection
        db_config = get_db_config()
        try:
            import mysql.connector
            test_conn = mysql.connector.connect(
                host=db_config.get('host'),
                user=db_config.get('user'),
                password=db_config.get('password'),
                database=db_config.get('database'),
                port=db_config.get('port', 3306),
                connect_timeout=10
            )
            test_conn.close()
            app.logger.info("Database connection verified")
        except Exception as db_error:
            return jsonify({
                'success': False, 
                'error': f'Database connection failed: {str(db_error)}'
            }), 500
        
        # Import and use generate_mock_data
        from generate_mock_data import generate_mock_data, insert_mock_data, connect_database
        
        # Generate data
        app.logger.info(f"Generating {num_records} mock records...")
        data = generate_mock_data(num_records)
        
        # Connect and insert
        connection = connect_database(db_config)
        if not connection:
            return jsonify({
                'success': False,
                'error': 'Failed to connect to database'
            }), 500
        
        app.logger.info("Inserting data into database...")
        try:
            success = insert_mock_data(connection, data, batch_size=50)
        except Exception as insert_error:
            app.logger.error(f"Insert error: {insert_error}")
            import traceback
            app.logger.error(traceback.format_exc())
            connection.close()
            return jsonify({
                'success': False,
                'error': f'Insertion failed: {str(insert_error)}. Check logs for details.'
            }), 500
        finally:
            connection.close()
        
        if success:
            # Count inserted records
            try:
                conn = mysql.connector.connect(
                    host=db_config.get('host'),
                    user=db_config.get('user'),
                    password=db_config.get('password'),
                    database=db_config.get('database'),
                    port=db_config.get('port', 3306),
                    connect_timeout=10
                )
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM application_forms WHERE project_cost_numeric IS NOT NULL AND lot_area IS NOT NULL AND lot_area > 0")
                total_records = cursor.fetchone()[0]
                cursor.close()
                conn.close()
            except:
                total_records = None
            
            return jsonify({
                'success': True,
                'message': f'Successfully inserted {len(data)} training records',
                'records_inserted': len(data),
                'total_records_in_db': total_records
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to insert data. Check logs for details.'
            }), 500
        
    except Exception as e:
        app.logger.error(f"Error inserting training data: {e}")
        import traceback
        error_trace = traceback.format_exc()
        app.logger.error(f"Full traceback: {error_trace}")
        return jsonify({
            'success': False,
            'error': f'Insertion failed: {str(e)}',
            'details': str(e) if len(str(e)) < 200 else str(e)[:200] + '...'
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
            db_host = db_config.get('host', 'localhost')
            
            # Check if using localhost (won't work on Heroku)
            if db_host == 'localhost' or db_host == '127.0.0.1':
                db_status = {
                    'connected': False,
                    'error': 'Cannot connect to localhost from Heroku. Need cloud database. Set DB_HOST to your remote database hostname.',
                    'host': db_host,
                    'database': db_config.get('database')
                }
            else:
                conn = mysql.connector.connect(
                    host=db_host,
                    user=db_config.get('user', 'root'),
                    password=db_config.get('password', ''),
                    database=db_config.get('database', 'u520834156_dbUPAHOZoning'),
                    port=db_config.get('port', 3306),
                    connect_timeout=10
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
                    'host': db_host,
                    'port': db_config.get('port', 3306),
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

