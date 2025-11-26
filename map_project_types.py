#!/usr/bin/env python3
"""
Map project types from application_forms to project_types table IDs
Updates application_time_series to use correct project_type_id
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

def create_type_mapping():
    """Create mapping between application_forms.project_type and project_types.id"""
    db_config = get_db_config()
    
    conn = mysql.connector.connect(
        host=db_config.get('host', 'localhost'),
        user=db_config.get('user', 'root'),
        password=db_config.get('password', ''),
        database=db_config.get('database', 'u520834156_dbUPAHOZoning'),
        charset='utf8mb4'
    )
    
    cursor = conn.cursor(dictionary=True)
    
    # Get all project types from project_types table
    cursor.execute("SELECT id, type_name, type_category FROM project_types WHERE is_active = 1")
    db_types = cursor.fetchall()
    
    # Create mapping dictionary
    type_mapping = {}
    for row in db_types:
        # Map by both type_name and type_category
        name_lower = row['type_name'].lower()
        category_lower = row['type_category'].lower() if row['type_category'] else None
        
        type_mapping[name_lower] = row['id']
        if category_lower and category_lower != name_lower:
            type_mapping[category_lower] = row['id']
    
    # Handle special cases
    special_mappings = {
        'residential-commercial': 'mixed_use',
        'mixed-use': 'mixed_use',
        'parks_recreational': 'other',
        'recreational': 'other',
        'aquaculture': 'agricultural',
        'mangrove': 'agricultural'
    }
    
    # Get distinct project types from application_forms
    cursor.execute("""
        SELECT DISTINCT project_type
        FROM application_forms
        WHERE project_type IS NOT NULL AND project_type != ''
    """)
    
    app_types = cursor.fetchall()
    
    print("Project Type Mapping:")
    print("=" * 60)
    
    for row in app_types:
        app_type = row['project_type'].lower()
        
        # Try direct mapping
        if app_type in type_mapping:
            mapped_id = type_mapping[app_type]
            print(f"  {row['project_type']} -> ID {mapped_id}")
        # Try special mapping
        elif app_type in special_mappings:
            mapped_category = special_mappings[app_type]
            # Find ID by category
            for db_type in db_types:
                if db_type['type_category'] == mapped_category:
                    mapped_id = db_type['id']
                    print(f"  {row['project_type']} -> ID {mapped_id} (via {mapped_category})")
                    break
        else:
            print(f"  {row['project_type']} -> NOT FOUND (needs manual mapping)")
    
    cursor.close()
    conn.close()
    
    return type_mapping, special_mappings

if __name__ == '__main__':
    create_type_mapping()

