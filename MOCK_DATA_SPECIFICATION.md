# Mock Data Specification

## Overview
The `generate_mock_data.py` script creates realistic training data for the land cost prediction model. It generates data for two tables: `clients` and `application_forms`.

## Generated Data Fields

### Application Forms Data

Each record includes the following fields:

#### 1. **Project Type** (Weighted Distribution)
- `residential` (35% - most common)
- `commercial` (20%)
- `industrial` (15%)
- `agricultural` (10%)
- `mixed-use` (8%)
- `institutional` (5%)
- `recreational` (4%)
- `residential-commercial` (3%)

#### 2. **Project Nature**
Randomly selected from:
- `new_construction`
- `renovation`
- `expansion`
- `conversion`
- `subdivision`
- `redevelopment`

#### 3. **Project Location**
Randomly selected from:
- `Barangay 1` through `Barangay 10`
- `Downtown`
- `Suburban`
- `Rural`
- `Urban Core`
- `Industrial Zone`
- `Commercial District`
- `Residential Area`
- `Mixed Zone`

#### 4. **Area Data**

**Lot Area** (varies by project type):
- Residential: 80 - 800 sqm
- Commercial: 100 - 2,000 sqm
- Industrial: 500 - 5,000 sqm
- Agricultural: 1,000 - 10,000 sqm
- Mixed-use: 200 - 3,000 sqm
- Institutional: 300 - 2,000 sqm
- Recreational: 500 - 5,000 sqm
- Residential-commercial: 150 - 1,500 sqm

**Project Area** (calculated as percentage of lot area):
- Agricultural: 50-100% of lot area
- Commercial/Industrial: 40-80% of lot area
- Residential/Others: 30-70% of lot area

#### 5. **Cost Data** (Realistic Pricing in PHP)

**Base Cost per sqm by Project Type:**
- Residential: 8,000 - 35,000 PHP/sqm
- Commercial: 20,000 - 60,000 PHP/sqm
- Industrial: 15,000 - 45,000 PHP/sqm
- Agricultural: 3,000 - 12,000 PHP/sqm
- Mixed-use: 18,000 - 50,000 PHP/sqm
- Institutional: 25,000 - 55,000 PHP/sqm
- Recreational: 12,000 - 40,000 PHP/sqm
- Residential-commercial: 15,000 - 45,000 PHP/sqm

**Cost Adjustments Applied:**
1. **Location Premium/Discount:**
   - Downtown/Urban Core: +15% to +35% premium
   - Rural: -10% to -25% discount
   - Other locations: Base price

2. **Area-Based Pricing (Economies of Scale):**
   - Large projects (>2000 sqm): -5% to -20% discount
   - Small projects (<150 sqm): +5% to +20% premium
   - Medium projects: Base price

3. **Year-Based Inflation:**
   - 5% increase per year (simulates price inflation over time)

4. **Seasonal Variation:**
   - Holiday season (Nov-Dec-Jan): +3% to +10%
   - Summer (Jun-Jul-Aug): -3% to +3%
   - Other months: Base price

**Final Calculation:**
```
cost_per_sqm = base_cost × location_factor × area_factor × year_factor × month_factor
project_cost_numeric = lot_area × cost_per_sqm
```

**Minimum Cost:** 100,000 PHP (enforced)

#### 6. **Time Data**
- **Created At:** Random date within last 3 years (1095 days)
- **Updated At:** Created at + 5 to 60 days

#### 7. **Status**
Randomly assigned:
- `approved`
- `pending`
- `rejected`

### Client Data

#### 1. **Client ID**
Format: `MOCK{6-digit-random}{record-index}`
Example: `MOCK1234560`

#### 2. **Personal Information**
- **First Name:** `Client{4-digit-random}`
  - Example: `Client5678`
- **Last Name:** `LastName{3-digit-random}`
  - Example: `LastName123`
- **Username:** `user_{client_id}_{4-digit-random}`
  - Example: `user_MOCK1234560_9876`
- **Email:** `{firstname}.{lastname}@example.com`
  - Example: `client5678.lastname123@example.com`
- **Cellphone:** `09{9-digit-random}`
  - Example: `09123456789`

#### 3. **Demographics**
- **Age:** 25-70 years old
  - 40% chance of being 35-55 (middle-aged focus)
- **Gender:** Randomly selected from:
  - `male`
  - `female`
  - `other`
- **Date of Birth:** Calculated from age (random month/day)
- **Age (stored as date):** Same as date_of_birth

## Data Distribution

### Time Distribution
- Records span **3 years** (1095 days)
- Uniformly distributed across the time period

### Project Type Distribution
- Weighted to reflect real-world distribution
- Residential projects are most common (35%)
- Agricultural and recreational are less common

### Cost Distribution
- Realistic cost ranges based on project type
- Includes location, size, and time-based variations
- Simulates real market conditions

## Example Record

```json
{
  "project_type": "residential",
  "project_nature": "new_construction",
  "project_location": "Downtown",
  "project_area": 450,
  "lot_area": 600,
  "project_cost_numeric": 18900000,
  "created_at": "2024-03-15T10:30:00",
  "age": 42,
  "gender": "male"
}
```

**Calculation Example:**
- Base cost: 20,000 PHP/sqm (residential)
- Location: Downtown (+25% premium) = 25,000 PHP/sqm
- Area: 600 sqm (medium, no adjustment) = 25,000 PHP/sqm
- Year: 2024 (1 year inflation +5%) = 26,250 PHP/sqm
- Month: March (no seasonal adjustment) = 26,250 PHP/sqm
- Total: 600 × 26,250 = 15,750,000 PHP
- Adjusted: 18,900,000 PHP (with variations)

## Database Tables Populated

### `clients` Table
- `id` (VARCHAR) - Unique client ID
- `firstname` (VARCHAR)
- `lastname` (VARCHAR)
- `date_of_birth` (DATE)
- `age` (DATE) - Stored as date
- `gender` (VARCHAR)
- `email` (VARCHAR)
- `cellphone` (VARCHAR)
- `username` (VARCHAR)
- `created_at` (DATETIME)
- `updated_at` (DATETIME)

### `application_forms` Table
- `project_type` (VARCHAR)
- `project_nature` (VARCHAR)
- `project_location` (VARCHAR)
- `project_area` (DECIMAL/FLOAT)
- `lot_area` (DECIMAL/FLOAT)
- `project_cost_numeric` (DECIMAL/FLOAT)
- `created_at` (DATETIME)
- `updated_at` (DATETIME)
- `client_id` (VARCHAR) - Foreign key to clients.id
- `status` (VARCHAR) - 'approved', 'pending', or 'rejected'

## Data Quality Features

1. **Realistic Relationships:**
   - Project area is always ≤ lot area
   - Cost correlates with project type and location
   - Area ranges match project type characteristics

2. **Temporal Consistency:**
   - Updated_at is always after created_at
   - Year-based inflation increases costs over time
   - Seasonal variations reflect market patterns

3. **Duplicate Handling:**
   - Uses `ON DUPLICATE KEY UPDATE` for application_forms
   - Checks for existing clients before inserting
   - Skips duplicates gracefully

4. **Batch Processing:**
   - Processes in batches of 50 records
   - Commits after each batch
   - Continues even if individual records fail

## Usage

```bash
# Insert 500 records (default)
curl https://your-app.herokuapp.com/api/insert_training_data

# Insert 1000 records
curl "https://your-app.herokuapp.com/api/insert_training_data?num_records=1000"

# Insert 2000 records
curl "https://your-app.herokuapp.com/api/insert_training_data?num_records=2000"
```

**Maximum:** 5000 records per request

## Training Data Requirements

For effective model training:
- **Minimum:** 50-100 records
- **Recommended:** 500-1000 records
- **Optimal:** 1000+ records

The generated data includes all necessary fields for training:
- ✅ `project_cost_numeric` (target variable)
- ✅ `lot_area` (feature)
- ✅ `project_area` (feature)
- ✅ `project_type` (feature)
- ✅ `project_location` (feature)
- ✅ Temporal data for time series analysis



