# SQL Mock Data Insertion Guide

## Overview
This guide explains how to use the `insert_mock_data.sql` script to populate your `application_forms` table with realistic mock data for training the land cost prediction model.

## Quick Start

### Option 1: Using MySQL Command Line
```bash
mysql -u u520834156_uPAHOZone25 -p u520834156_dbUPAHOZoning < insert_mock_data.sql
```

### Option 2: Using MySQL Workbench
1. Open MySQL Workbench
2. Connect to your database
3. File → Open SQL Script → Select `insert_mock_data.sql`
4. Execute the script (Ctrl+Shift+Enter)

### Option 3: Using phpMyAdmin
1. Log into phpMyAdmin
2. Select your database: `u520834156_dbUPAHOZoning`
3. Click "SQL" tab
4. Copy-paste the contents of `insert_mock_data.sql`
5. Click "Go"

## What the Script Does

### 1. Manual INSERT Statements (3 Sample Records)
- Shows the structure of data being inserted
- Useful for understanding the format
- Can be duplicated for more records

### 2. Stored Procedure: `InsertMockApplicationForms`
- Efficiently inserts large batches of data
- Generates realistic random data
- Handles all new columns from your ALTER TABLE

### 3. Verification Queries
- Counts total records inserted
- Shows statistics (avg cost, status distribution)
- Displays sample records

## Data Generated

### Application Numbers
Format: `APP-YYYY-XXXXXX`
Example: `APP-2025-000001`

### Applicant Names
Random combinations of common Filipino names:
- First names: Juan, Maria, Roberto, Ana, Carlos, Elena, etc.
- Last names: Dela Cruz, Santos, Garcia, Reyes, Lopez, etc.

### Project Types (Weighted Distribution)
- **Residential** (35%) - Most common
- **Commercial** (20%)
- **Industrial** (15%)
- **Agricultural** (10%)
- **Mixed-use** (8%)
- **Institutional** (5%)
- **Recreational** (4%)
- **Residential-commercial** (3%)

### Project Nature (ENUM)
- `new_development` (33%)
- `improvement` (33%)
- `others_renovation` (33%)

### Areas (varies by project type)
- **Residential**: 80-800 sqm
- **Commercial**: 100-2,000 sqm
- **Industrial**: 500-5,000 sqm
- **Agricultural**: 1,000-10,000 sqm
- **Mixed-use**: 200-3,000 sqm
- **Institutional**: 300-2,000 sqm
- **Recreational**: 500-5,000 sqm

### Cost Calculation
- Base cost per sqm by project type
- Location premium/discount:
  - Downtown/Urban Core: +15% to +35%
  - Rural: -10% to -25%
- Area-based pricing:
  - Large projects (>2000 sqm): -5% to -20%
  - Small projects (<150 sqm): +5% to +20%
- Minimum cost: 100,000 PHP

### Other Fields
- **Land Right**: 80% owner, 15% lessee, 5% others
- **Project Tenure**: 90% permanent, 10% temporary
- **Written Notice Subject**: 70% yes, 30% no
- **Release Mode**: 60% pickup, 40% mail
- **Status**: 33% approved, 33% pending, 33% rejected
- **Dates**: Random within last 3 years

## Customizing the Script

### Change Number of Records
Edit this line in the script:
```sql
CALL InsertMockApplicationForms(1000);  -- Change 1000 to your desired number
```

### Adjust Project Type Distribution
Modify the CASE statement in the stored procedure:
```sql
SET proj_type = CASE
    WHEN RAND() < 0.35 THEN 'residential'  -- Adjust percentages
    WHEN RAND() < 0.55 THEN 'commercial'
    ...
END;
```

### Modify Cost Ranges
Edit the cost calculation section:
```sql
SET cost_fig = CASE proj_type
    WHEN 'residential' THEN lot_val * (8000 + RAND() * 27000)  -- Adjust ranges
    ...
END;
```

## Verification

After running the script, verify the data:

```sql
-- Count records
SELECT COUNT(*) FROM application_forms 
WHERE application_number LIKE CONCAT('APP-', YEAR(NOW()), '-%');

-- Check cost distribution
SELECT 
    project_type,
    COUNT(*) as count,
    AVG(project_cost_figures) as avg_cost,
    MIN(project_cost_figures) as min_cost,
    MAX(project_cost_figures) as max_cost
FROM application_forms
WHERE application_number LIKE CONCAT('APP-', YEAR(NOW()), '-%')
GROUP BY project_type;

-- Check status distribution
SELECT status, COUNT(*) as count
FROM application_forms
WHERE application_number LIKE CONCAT('APP-', YEAR(NOW()), '-%')
GROUP BY status;
```

## Troubleshooting

### Error: "Column doesn't exist"
- Make sure you've run the ALTER TABLE statement first
- Check that all columns from the ALTER TABLE are present

### Error: "Duplicate entry for key 'application_number'"
- The script generates unique application numbers
- If you run it multiple times, change the year or add a prefix

### Error: "Data too long for column"
- Some generated text might be too long
- Adjust VARCHAR lengths in the ALTER TABLE if needed

### Performance Issues
- For large datasets (>5000 records), run in smaller batches
- Disable indexes temporarily, then rebuild them

## Example Output

After running the script, you should see:
```
Query OK, 1000 rows affected (X.XX sec)
```

And verification query results:
```
total_records: 1000
unique_applications: 1000
earliest_date: 2022-11-26
latest_date: 2025-11-26
avg_cost: 25,450,000.00
approved_count: 333
pending_count: 334
rejected_count: 333
```

## Next Steps

1. ✅ Run the ALTER TABLE statement (if not done)
2. ✅ Run `insert_mock_data.sql`
3. ✅ Verify data with queries
4. ✅ Train models using `/api/train` endpoint
5. ✅ Test predictions

## Notes

- The script preserves existing data (only inserts new records)
- Application numbers are unique per year
- All dates are within the last 3 years
- Costs are realistic based on project type and location
- The script is idempotent (can be run multiple times safely)



