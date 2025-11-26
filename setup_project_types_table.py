#!/usr/bin/env python3
"""
Create or update project_types table to match the IDs used in application_time_series
"""

import mysql.connector
import json
import os

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

def setup_project_types_table():
    """Create or update project_types table"""
    db_config = get_db_config()
    
    try:
        conn = mysql.connector.connect(
            host=db_config.get('host', 'localhost'),
            user=db_config.get('user', 'root'),
            password=db_config.get('password', ''),
            database=db_config.get('database', 'u520834156_dbUPAHOZoning'),
            charset='utf8mb4'
        )
        
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SHOW TABLES LIKE 'project_types'")
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            # Create table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS project_types (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    name VARCHAR(100) UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            print("Created project_types table")
        else:
            print("project_types table already exists")
        
        # Get distinct project types from application_forms
        cursor.execute("""
            SELECT DISTINCT project_type
            FROM application_forms
            WHERE project_type IS NOT NULL AND project_type != ''
            ORDER BY project_type ASC
        """)
        
        project_types = cursor.fetchall()
        
        # Insert or update project types with sequential IDs
        for idx, (project_type,) in enumerate(project_types, 1):
            cursor.execute("""
                INSERT INTO project_types (id, name)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE name = VALUES(name)
            """, (idx, project_type))
        
        conn.commit()
        
        # Verify
        cursor.execute("SELECT id, name FROM project_types ORDER BY id")
        results = cursor.fetchall()
        print(f"\nProject types in table ({len(results)} total):")
        for id, name in results:
            print(f"  ID {id}: {name}")
        
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
    print("Setting up project_types table")
    print("=" * 60)
    setup_project_types_table()

