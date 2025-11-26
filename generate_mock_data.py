#!/usr/bin/env python3
"""
Generate Mock Data for Land Cost Prediction Model Training
Creates realistic sample data in the database
"""

import mysql.connector
import random
from datetime import datetime, timedelta
import json
import os

# Database configuration
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

def connect_database(db_config):
    """Connect to MySQL database"""
    try:
        connection = mysql.connector.connect(
            host=db_config.get('host', 'localhost'),
            user=db_config.get('user', 'root'),
            password=db_config.get('password', ''),
            database=db_config.get('database', 'u520834156_dbUPAHOZoning'),
            charset='utf8mb4',
            connect_timeout=10
        )
        return connection
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def generate_mock_data(num_records=1500):
    """Generate comprehensive mock data for training"""
    
    # Project types and their typical cost ranges (per sqm in PHP)
    project_types = [
        'residential', 'commercial', 'industrial', 'agricultural', 
        'mixed-use', 'institutional', 'recreational', 'residential-commercial'
    ]
    
    # Weight project types for more realistic distribution
    project_type_weights = {
        'residential': 0.35,  # Most common
        'commercial': 0.20,
        'industrial': 0.15,
        'agricultural': 0.10,
        'mixed-use': 0.08,
        'institutional': 0.05,
        'recreational': 0.04,
        'residential-commercial': 0.03
    }
    
    project_natures = [
        'new_construction', 'renovation', 'expansion', 'conversion', 
        'subdivision', 'redevelopment'
    ]
    
    locations = [
        'Barangay 1', 'Barangay 2', 'Barangay 3', 'Barangay 4', 'Barangay 5',
        'Barangay 6', 'Barangay 7', 'Barangay 8', 'Barangay 9', 'Barangay 10',
        'Downtown', 'Suburban', 'Rural', 'Urban Core', 'Industrial Zone',
        'Commercial District', 'Residential Area', 'Mixed Zone'
    ]
    
    genders = ['male', 'female', 'other']
    
    # Cost per sqm ranges by project type (in PHP) - more realistic ranges
    cost_ranges = {
        'residential': (8000, 35000),
        'commercial': (20000, 60000),
        'industrial': (15000, 45000),
        'agricultural': (3000, 12000),
        'mixed-use': (18000, 50000),
        'institutional': (25000, 55000),
        'recreational': (12000, 40000),
        'residential-commercial': (15000, 45000)
    }
    
    # Area ranges (in sqm) - more diverse
    lot_area_ranges_by_type = {
        'residential': (80, 800),
        'commercial': (100, 2000),
        'industrial': (500, 5000),
        'agricultural': (1000, 10000),
        'mixed-use': (200, 3000),
        'institutional': (300, 2000),
        'recreational': (500, 5000),
        'residential-commercial': (150, 1500)
    }
    
    # Age range
    age_range = (25, 70)
    
    # Generate data
    data = []
    start_date = datetime.now() - timedelta(days=1095)  # 3 years ago for more time diversity
    
    # Create weighted project type list
    weighted_project_types = []
    for ptype, weight in project_type_weights.items():
        weighted_project_types.extend([ptype] * int(weight * 100))
    
    for i in range(num_records):
        # Random date within last 3 years
        days_ago = random.randint(0, 1095)
        created_at = start_date + timedelta(days=days_ago)
        
        # Select project type based on weights
        project_type = random.choice(weighted_project_types)
        
        # Generate lot area based on project type
        lot_area_range = lot_area_ranges_by_type[project_type]
        lot_area = random.randint(*lot_area_range)
        
        # Project area is typically 30-90% of lot area, but varies by type
        if project_type == 'agricultural':
            project_area_ratio = random.uniform(0.5, 1.0)  # Agricultural uses more of the lot
        elif project_type in ['commercial', 'industrial']:
            project_area_ratio = random.uniform(0.4, 0.8)  # Commercial/industrial buildings
        else:
            project_area_ratio = random.uniform(0.3, 0.7)  # Residential and others
        
        project_area = int(lot_area * project_area_ratio)
        
        # Generate cost based on project type, area, and location
        base_cost_per_sqm = random.uniform(*cost_ranges[project_type])
        
        # Location premium/discount
        location_factor = 1.0
        if 'Downtown' in locations or 'Urban Core' in locations:
            location_factor = random.uniform(1.15, 1.35)  # Premium locations
        elif 'Rural' in locations:
            location_factor = random.uniform(0.75, 0.90)  # Lower cost areas
        
        # Area-based pricing (economies of scale)
        area_factor = 1.0
        if lot_area > 2000:
            area_factor = random.uniform(0.80, 0.95)  # Large projects get discount
        elif lot_area < 150:
            area_factor = random.uniform(1.05, 1.20)  # Small projects pay premium
        
        # Year-based inflation (prices increase over time)
        year_factor = 1.0 + ((created_at.year - start_date.year) * 0.05)  # 5% per year
        
        # Seasonal variation
        month_factor = 1.0
        if created_at.month in [11, 12, 1]:  # Holiday season
            month_factor = random.uniform(1.03, 1.10)
        elif created_at.month in [6, 7, 8]:  # Summer
            month_factor = random.uniform(0.97, 1.03)
        
        # Calculate final cost per sqm
        cost_per_sqm = base_cost_per_sqm * location_factor * area_factor * year_factor * month_factor
        
        # Total project cost
        project_cost_numeric = int(lot_area * cost_per_sqm)
        
        # Ensure minimum cost
        if project_cost_numeric < 100000:
            project_cost_numeric = random.randint(100000, 500000)
        
        # Generate client data with realistic distribution
        age = random.randint(*age_range)
        # Slightly more middle-aged applicants
        if random.random() < 0.4:
            age = random.randint(35, 55)
        
        gender = random.choice(genders)
        
        record = {
            'project_type': project_type,
            'project_nature': random.choice(project_natures),
            'project_location': random.choice(locations),
            'project_area': project_area,
            'lot_area': lot_area,
            'project_cost_numeric': project_cost_numeric,
            'created_at': created_at,
            'age': age,
            'gender': gender
        }
        
        data.append(record)
    
    return data

def insert_mock_data(connection, data):
    """Insert mock data into database"""
    cursor = connection.cursor()
    
    inserted_clients = 0
    inserted_applications = 0
    
    try:
        for i, record in enumerate(data):
            # Generate unique client ID
            client_id = f"MOCK{random.randint(100000, 999999)}{i}"
            
            # Calculate date of birth from age
            birth_year = datetime.now().year - record['age']
            birth_month = random.randint(1, 12)
            birth_day = random.randint(1, 28)
            date_of_birth = datetime(birth_year, birth_month, birth_day).date()
            
            # Age as date (this seems to be stored as date in the database)
            age_date = date_of_birth
            
            # Check if client already exists
            cursor.execute("SELECT id FROM clients WHERE id = %s", (client_id,))
            if cursor.fetchone():
                # Client exists, use it
                pass
            else:
                # Insert new client
                firstname = f"Client{random.randint(1000, 9999)}"
                lastname = f"LastName{random.randint(100, 999)}"
                
                username = f"user_{client_id}_{random.randint(1000, 9999)}"
                email = f"{firstname.lower()}.{lastname.lower()}@example.com"
                
                cursor.execute("""
                    INSERT INTO clients 
                    (id, firstname, lastname, date_of_birth, age, gender, 
                     email, cellphone, username, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    client_id,
                    firstname,
                    lastname,
                    date_of_birth,
                    age_date,  # Age stored as date
                    record['gender'],
                    email,
                    f"09{random.randint(100000000, 999999999)}",
                    username,
                    record['created_at'],
                    record['created_at']
                ))
                inserted_clients += 1
            
            # Insert application form
            cursor.execute("""
                INSERT INTO application_forms 
                (project_type, project_nature, project_location, project_area, 
                 lot_area, project_cost_numeric, created_at, updated_at, client_id, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                record['project_type'],
                record['project_nature'],
                record['project_location'],
                record['project_area'],
                record['lot_area'],
                record['project_cost_numeric'],
                record['created_at'],
                record['created_at'] + timedelta(days=random.randint(5, 60)),
                client_id,
                random.choice(['approved', 'pending', 'rejected'])
            ))
            inserted_applications += 1
        
        connection.commit()
        print(f"Successfully inserted {inserted_clients} clients and {inserted_applications} applications")
        return True
        
    except Exception as e:
        connection.rollback()
        print(f"Error inserting data: {e}")
        return False
    finally:
        cursor.close()

def main():
    """Main function"""
    print("=" * 60)
    print("Generating Mock Data for Land Cost Prediction")
    print("=" * 60)
    
    # Get database config
    db_config = get_db_config()
    
    # Connect to database
    print("\nConnecting to database...")
    connection = connect_database(db_config)
    if not connection:
        print("Failed to connect to database")
        return
    
    # Generate mock data
    print("\nGenerating mock data...")
    num_records = 1500  # Generate 1500 records for better model training
    data = generate_mock_data(num_records)
    print(f"Generated {len(data)} mock records")
    
    # Show data distribution
    project_type_counts = {}
    for record in data:
        pt = record['project_type']
        project_type_counts[pt] = project_type_counts.get(pt, 0) + 1
    
    print("\nData Distribution by Project Type:")
    for pt, count in sorted(project_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pt}: {count} records ({count/len(data)*100:.1f}%)")
    
    # Insert data
    print("\nInserting data into database...")
    success = insert_mock_data(connection, data)
    
    if success:
        print("\n" + "=" * 60)
        print("Mock data generation completed successfully!")
        print("=" * 60)
        print(f"\nYou can now train the model with {len(data)} records")
    else:
        print("\nFailed to insert mock data")
    
    connection.close()

if __name__ == '__main__':
    main()

