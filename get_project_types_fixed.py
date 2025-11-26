#!/usr/bin/env python3
"""
Get Project Types with correct IDs from project_types table
For PHP interface
"""

import mysql.connector
import json
import os
import sys

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

def get_project_types():
    """Get project types from project_types table"""
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
            WHERE pt.is_active = 1
            ORDER BY pt.type_name ASC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Also check for project types that might not match exactly
        cursor.execute("""
            SELECT DISTINCT project_type 
            FROM application_forms 
            WHERE project_type IS NOT NULL AND project_type != ''
        """)
        app_types = [r['project_type'] for r in cursor.fetchall()]
        
        # Create mapping
        project_types = []
        for row in results:
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
        
        # Add any missing types from application_forms
        for app_type in app_types:
            found = False
            for pt in project_types:
                if app_type.lower() in [pt['name'].lower(), pt.get('category', '').lower()]:
                    found = True
                    break
            
            if not found:
                # Try to find a match
                cursor.execute("""
                    SELECT id, type_name, type_category 
                    FROM project_types 
                    WHERE LOWER(type_name) LIKE %s 
                    OR LOWER(type_category) LIKE %s
                    LIMIT 1
                """, (f"%{app_type.lower()}%", f"%{app_type.lower()}%"))
                match = cursor.fetchone()
                
                if match:
                    project_types.append({
                        'id': match['id'],
                        'name': match['type_name'],
                        'category': match['type_category'],
                        'has_time_series_data': False,
                        'time_series_count': 0
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
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        try:
            with open(input_file, 'r') as f:
                request = json.load(f)
        except:
            request = {}
    else:
        request = {}
    
    result = get_project_types()
    print(json.dumps(result, indent=2))

