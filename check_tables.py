#!/usr/bin/env python3
"""Check database tables for SARIMAX"""

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

db_config = get_db_config()
conn = mysql.connector.connect(
    host=db_config.get('host', 'localhost'),
    user=db_config.get('user', 'root'),
    password=db_config.get('password', ''),
    database=db_config.get('database', 'u520834156_dbUPAHOZoning')
)

cursor = conn.cursor()
cursor.execute('SHOW TABLES')
tables = [t[0] for t in cursor.fetchall()]

print('Checking for SARIMAX-related tables:')
has_time_series = 'application_time_series' in tables
has_predictions = 'sarimax_predictions' in tables
print(f'\napplication_time_series exists: {has_time_series}')
print(f'sarimax_predictions exists: {has_predictions}')

if 'application_time_series' in tables:
    cursor.execute('DESCRIBE application_time_series')
    cols = cursor.fetchall()
    print('\napplication_time_series columns:')
    for col in cols:
        print(f'  - {col[0]} ({col[1]})')
    
    cursor.execute('SELECT COUNT(*) FROM application_time_series')
    count = cursor.fetchone()[0]
    print(f'\nRecords in application_time_series: {count}')

conn.close()

