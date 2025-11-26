#!/usr/bin/env python3
"""
Get Project Types from Database
Returns JSON list of available project types for SARIMAX interface
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
    """Get all distinct project types from database"""
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
        
        # Get distinct project types with counts
        query = """
            SELECT 
                project_type,
                COUNT(*) as count
            FROM application_forms
            WHERE project_type IS NOT NULL
            AND project_type != ''
            GROUP BY project_type
            ORDER BY project_type ASC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Format as list of objects with id and name
        project_types = []
        for idx, row in enumerate(results, 1):
            project_types.append({
                'id': idx,
                'name': row['project_type'],
                'count': row['count']
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
    result = get_project_types()
    print(json.dumps(result, indent=2))

