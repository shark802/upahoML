#!/usr/bin/env python3
"""
Populate application_time_series table from application_forms
This creates monthly aggregated data needed for SARIMAX predictions
"""

import mysql.connector
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

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

def get_project_type_mapping(conn):
    """Get or create mapping between project_type names and IDs"""
    cursor = conn.cursor(dictionary=True)
    
    # Get distinct project types
    cursor.execute("""
        SELECT DISTINCT project_type 
        FROM application_forms 
        WHERE project_type IS NOT NULL AND project_type != ''
        ORDER BY project_type
    """)
    
    project_types = cursor.fetchall()
    
    # Create mapping (using index as ID, starting from 1)
    type_mapping = {}
    for idx, row in enumerate(project_types, 1):
        type_mapping[row['project_type']] = idx
    
    cursor.close()
    return type_mapping

def populate_time_series():
    """Populate application_time_series table"""
    db_config = get_db_config()
    
    try:
        conn = mysql.connector.connect(
            host=db_config.get('host', 'localhost'),
            user=db_config.get('user', 'root'),
            password=db_config.get('password', ''),
            database=db_config.get('database', 'u520834156_dbUPAHOZoning'),
            charset='utf8mb4'
        )
        
        # Get project type mapping
        type_mapping = get_project_type_mapping(conn)
        print(f"Found {len(type_mapping)} project types")
        
        cursor = conn.cursor()
        
        # Get all applications grouped by month and project type
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
        
        print(f"Found {len(results)} month-project_type combinations")
        
        # Insert or update time series data
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
        
        inserted = 0
        updated = 0
        
        for row in results:
            month_date, year, month, project_type, app_count, approved, rejected, pending, total_area, avg_area, total_cost, avg_cost = row
            
            project_type_id = type_mapping.get(project_type)
            if not project_type_id:
                continue
            
            # Calculate approval rate
            approval_rate = (approved / app_count * 100) if app_count > 0 else 0
            
            try:
                cursor.execute(insert_query, (
                    month_date, year, month, project_type_id, app_count, approved,
                    rejected, pending, total_area, avg_area, total_cost, avg_cost, approval_rate
                ))
                if cursor.rowcount == 1:
                    inserted += 1
                else:
                    updated += 1
            except Exception as e:
                print(f"Error inserting {month_date}, {project_type}: {e}")
                continue
        
        conn.commit()
        print(f"\nCompleted: {inserted} inserted, {updated} updated")
        print(f"Total records in application_time_series: {cursor.rowcount}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        if conn:
            conn.close()
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("Populating application_time_series table")
    print("=" * 60)
    populate_time_series()

