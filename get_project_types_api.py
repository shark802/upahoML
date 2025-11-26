#!/usr/bin/env python3
"""
API to get project types for PHP interface
Returns project types with IDs that match application_time_series.project_type_id
"""

import mysql.connector
import json
import os
import sys
import warnings

# Suppress all warnings to ensure clean JSON output
warnings.filterwarnings('ignore')

# Redirect stderr to prevent any error messages from appearing in output
import io
sys.stderr = io.StringIO()

def get_db_config():
    """Get database configuration"""
    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        return {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'u520834156_dbUPAHOZoning'
        }

def get_project_types_with_ids():
    """Get project types with their corresponding IDs from project_types table"""
    db_config = get_db_config()
    
    try:
        conn = mysql.connector.connect(
            host=db_config.get('host', 'localhost'),
            user=db_config.get('user', 'root'),
            password=db_config.get('password', ''),
            database=db_config.get('database', 'u520834156_dbUPAHOZoning'),
            charset='utf8mb4'
        )
        
        cursor = conn.cursor(dictionary=True)
        
        # Get project types that have data in application_forms
        query = """
            SELECT DISTINCT pt.id, pt.type_name, pt.type_category
            FROM project_types pt
            INNER JOIN application_forms af ON LOWER(af.project_type) = LOWER(pt.type_name) 
                OR LOWER(af.project_type) = LOWER(pt.type_category)
                OR (af.project_type = 'mixed-use' AND pt.type_category = 'mixed_use')
                OR (af.project_type = 'residential-commercial' AND pt.type_category = 'mixed_use')
            WHERE pt.is_active = 1
            ORDER BY pt.type_name ASC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        project_types = []
        seen_ids = set()
        
        for row in results:
            if row['id'] in seen_ids:
                continue
            seen_ids.add(row['id'])
            
            # Check if has time series data
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM application_time_series 
                WHERE project_type_id = %s
            """, (row['id'],))
            ts_count = cursor.fetchone()['count']
            
            project_types.append({
                'id': row['id'],
                'name': row['type_name'],
                'category': row['type_category'],
                'has_time_series_data': ts_count > 0,
                'time_series_count': ts_count
            })
        
        cursor.close()
        conn.close()
        
        return {
            'success': True,
            'project_types': project_types,
            'total': len(project_types)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'project_types': []
        }

if __name__ == '__main__':
    try:
        # Check if called with input file (for PHP compatibility)
        if len(sys.argv) > 1:
            input_file = sys.argv[1]
            try:
                with open(input_file, 'r') as f:
                    request = json.load(f)
            except:
                request = {}
        else:
            request = {}
        
        result = get_project_types_with_ids()
        # Output only JSON, no extra whitespace or errors
        print(json.dumps(result))
        sys.exit(0)
    except Exception as e:
        # Always return valid JSON even on errors
        error_result = {
            'success': False,
            'error': str(e),
            'project_types': []
        }
        print(json.dumps(error_result))
        sys.exit(1)

