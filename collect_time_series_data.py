#!/usr/bin/env python3
"""
Collect and populate time series data for SARIMAX
Called from PHP to collect/update application_time_series data
"""

import mysql.connector
import json
import os
import sys
import warnings
from datetime import datetime

# Suppress all warnings
warnings.filterwarnings('ignore')

# Redirect stderr
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

def collect_time_series_data():
    """Collect and populate time series data"""
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
        
        # Get project type mapping from project_types table
        cursor.execute("""
            SELECT id, type_name, type_category 
            FROM project_types 
            WHERE is_active = 1
        """)
        project_types = {row['id']: row for row in cursor.fetchall()}
        
        # Create mapping from application_forms project_type to project_types.id
        type_mapping = {}
        for pt_id, pt_data in project_types.items():
            type_mapping[pt_data['type_name'].lower()] = pt_id
            if pt_data['type_category']:
                type_mapping[pt_data['type_category'].lower()] = pt_id
        
        # Special mappings
        type_mapping['mixed-use'] = type_mapping.get('mixed_use', None)
        type_mapping['residential-commercial'] = type_mapping.get('mixed_use', None)
        
        # Get aggregated data by month and project type
        query = """
            SELECT 
                DATE_FORMAT(created_at, '%Y-%m-01') as month_date,
                YEAR(created_at) as year,
                MONTH(created_at) as month,
                project_type,
                COUNT(*) as application_count,
                SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) as approved_count,
                SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected_count,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_count,
                SUM(COALESCE(lot_area, 0)) as total_area_sqm,
                AVG(COALESCE(lot_area, 0)) as avg_area_sqm,
                SUM(COALESCE(project_cost_numeric, 0)) as total_cost,
                AVG(COALESCE(project_cost_numeric, 0)) as avg_cost
            FROM application_forms
            WHERE project_type IS NOT NULL 
            AND project_type != ''
            AND created_at IS NOT NULL
            GROUP BY DATE_FORMAT(created_at, '%Y-%m-01'), project_type
            ORDER BY month_date, project_type
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        inserted = 0
        updated = 0
        
        insert_query = """
            INSERT INTO application_time_series 
            (date, year, month, project_type_id, application_count, approved_count, 
             rejected_count, pending_count, total_area_sqm, avg_area_sqm, 
             total_cost, avg_cost, approval_rate, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
            application_count = VALUES(application_count),
            approved_count = VALUES(approved_count),
            rejected_count = VALUES(rejected_count),
            pending_count = VALUES(pending_count),
            total_area_sqm = VALUES(total_area_sqm),
            avg_area_sqm = VALUES(avg_area_sqm),
            total_cost = VALUES(total_cost),
            avg_cost = VALUES(avg_cost),
            approval_rate = VALUES(approval_rate),
            updated_at = NOW()
        """
        
        for row in results:
            project_type = row['project_type'].lower()
            project_type_id = type_mapping.get(project_type)
            
            if not project_type_id:
                continue
            
            approval_rate = (row['approved_count'] / row['application_count'] * 100) if row['application_count'] > 0 else 0
            
            try:
                cursor.execute(insert_query, (
                    row['month_date'], row['year'], row['month'], project_type_id,
                    row['application_count'], row['approved_count'], row['rejected_count'],
                    row['pending_count'], row['total_area_sqm'], row['avg_area_sqm'],
                    row['total_cost'], row['avg_cost'], approval_rate
                ))
                
                if cursor.rowcount == 1:
                    inserted += 1
                else:
                    updated += 1
            except Exception as e:
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            'success': True,
            'message': 'Time series data collected successfully',
            'inserted': inserted,
            'updated': updated,
            'total_records': len(results)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to collect time series data'
        }

if __name__ == '__main__':
    try:
        result = collect_time_series_data()
        print(json.dumps(result))
        sys.exit(0 if result['success'] else 1)
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'message': 'Failed to collect data'
        }
        print(json.dumps(error_result))
        sys.exit(1)

