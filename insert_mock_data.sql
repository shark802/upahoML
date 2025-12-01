-- =====================================================
-- SQL Script: Insert Mock Data for Application Forms
-- Compatible with the actual table structure
-- =====================================================
-- 
-- Usage:
--   1. Connect to your MySQL database
--   2. Run this script: mysql -u username -p database_name < insert_mock_data.sql
--   3. Or copy-paste into MySQL Workbench/phpMyAdmin
--
-- This script inserts 1000 mock records with realistic data
-- =====================================================

-- Disable foreign key checks temporarily for faster insertion
SET FOREIGN_KEY_CHECKS = 0;

-- =====================================================
-- INSERT MOCK DATA (3 Sample Records)
-- =====================================================

INSERT INTO application_forms (
    firstname,
    lastname,
    name_of_representative,
    applicant_address,
    representative_address,
    project_location,
    location_type,
    project_type,
    project_area,
    project_lifespan,
    project_significance,
    project_classification,
    site_zoning,
    land_uses,
    applicant_name,
    representative_name,
    project_area_sqm,
    project_nature,
    right_over_land,
    project_tenure,
    lot_area,
    building_improvement_area,
    existing_land_use_site,
    project_cost_numeric,
    project_cost_words,
    is_subject_of_notice,
    notice_offices,
    preferred_mode_of_release,
    address_to,
    existing_land_uses,
    written_notices,
    other_hsrc_offices,
    dates_filed,
    actions_taken,
    applicant_signature,
    representative_signature,
    longitude,
    latitude,
    status,
    created_at,
    updated_at
) VALUES
-- Record 1
(
    'Juan',
    'Dela Cruz',
    NULL,
    '123 Main Street, Barangay 1, City',
    NULL,
    'Barangay 1',
    1,
    'residential',
    450.00,
    'permanent',
    'local',
    'Residential Development',
    'R-1',
    'Residential housing',
    'Juan Dela Cruz',
    NULL,
    450.00,
    'new_development',
    'owner',
    'permanent',
    600.00,
    450.00,
    'Residential housing',
    18900000.00,
    'Eighteen Million Nine Hundred Thousand Pesos',
    'yes',
    'City Planning Office, Building Official',
    'pick_up',
    'applicant',
    'Residential housing',
    'yes',
    'City Planning Office, Building Official',
    '2024-01-15, 2024-02-20',
    'Initial review completed, Site inspection scheduled',
    'Juan Dela Cruz',
    NULL,
    '120.9842',
    '14.6042',
    'approved',
    DATE_SUB(NOW(), INTERVAL 90 DAY),
    DATE_SUB(NOW(), INTERVAL 60 DAY)
),
-- Record 2
(
    'Maria',
    'Santos',
    NULL,
    '456 Commercial Avenue, Downtown',
    NULL,
    'Downtown',
    1,
    'commercial',
    800.00,
    'permanent',
    'local',
    'Commercial Development',
    'C-2',
    'Commercial retail space',
    'Maria Santos',
    NULL,
    800.00,
    'improvement',
    'owner',
    'permanent',
    1200.00,
    800.00,
    'Commercial retail space',
    50000000.00,
    'Fifty Million Pesos',
    'yes',
    'Business Permit Office',
    'mail',
    'applicant',
    'Commercial retail space',
    'yes',
    'Business Permit Office',
    '2024-02-10',
    'Application under review',
    'Maria Santos',
    NULL,
    '120.9850',
    '14.6050',
    'pending',
    DATE_SUB(NOW(), INTERVAL 45 DAY),
    DATE_SUB(NOW(), INTERVAL 20 DAY)
),
-- Record 3
(
    'Roberto',
    'Garcia',
    'Pedro Garcia',
    '789 Industrial Road, Industrial Zone',
    '789 Industrial Road, Industrial Zone',
    'Industrial Zone',
    1,
    'industrial',
    2000.00,
    'permanent',
    'local',
    'Industrial Development',
    'I-1',
    'Industrial warehouse',
    'Roberto Garcia',
    'Pedro Garcia',
    2000.00,
    'new_development',
    'lessee',
    'permanent',
    3000.00,
    2000.00,
    'Industrial warehouse',
    120000000.00,
    'One Hundred Twenty Million Pesos',
    'no',
    NULL,
    'pick_up',
    'authorized_representative',
    'Industrial warehouse',
    'no',
    NULL,
    '2024-03-05',
    'Site inspection completed',
    'Roberto Garcia',
    'Pedro Garcia',
    '120.9860',
    '14.6060',
    'approved',
    DATE_SUB(NOW(), INTERVAL 60 DAY),
    DATE_SUB(NOW(), INTERVAL 15 DAY)
);

-- =====================================================
-- BULK INSERT USING STORED PROCEDURE
-- (More efficient for large datasets)
-- =====================================================

DELIMITER $$

DROP PROCEDURE IF EXISTS InsertMockApplicationForms$$

CREATE PROCEDURE InsertMockApplicationForms(IN num_records INT)
BEGIN
    DECLARE i INT DEFAULT 1;
    DECLARE first_name VARCHAR(100);
    DECLARE last_name VARCHAR(100);
    DECLARE app_name VARCHAR(200);
    DECLARE app_address TEXT;
    DECLARE proj_type VARCHAR(200);
    DECLARE proj_nature VARCHAR(250);
    DECLARE proj_location TEXT;
    DECLARE area_val DECIMAL(10,2);
    DECLARE lot_val DECIMAL(10,2);
    DECLARE building_val DECIMAL(10,2);
    DECLARE land_right_val VARCHAR(250);
    DECLARE tenure_val VARCHAR(250);
    DECLARE cost_fig DECIMAL(15,2);
    DECLARE cost_words VARCHAR(255);
    DECLARE notice_subject VARCHAR(10);
    DECLARE release_mode VARCHAR(255);
    DECLARE release_to VARCHAR(255);
    DECLARE created_date DATETIME;
    DECLARE updated_date DATETIME;
    DECLARE status_val VARCHAR(50);
    DECLARE longitude_val VARCHAR(255);
    DECLARE latitude_val VARCHAR(255);
    
    WHILE i <= num_records DO
        -- Generate names
        SET first_name = ELT(1 + FLOOR(RAND() * 20), 'Juan', 'Maria', 'Roberto', 'Ana', 'Carlos', 'Elena', 'Miguel', 'Carmen', 'Jose', 'Rosa', 'Pedro', 'Isabel', 'Fernando', 'Lucia', 'Antonio', 'Patricia', 'Manuel', 'Sofia', 'Ricardo', 'Gabriela');
        SET last_name = ELT(1 + FLOOR(RAND() * 20), 'Dela Cruz', 'Santos', 'Garcia', 'Reyes', 'Lopez', 'Gonzalez', 'Rodriguez', 'Fernandez', 'Martinez', 'Torres', 'Rivera', 'Ramos', 'Cruz', 'Morales', 'Ortiz', 'Gutierrez', 'Chavez', 'Jimenez', 'Mendoza', 'Vargas');
        SET app_name = CONCAT(first_name, ' ', last_name);
        
        -- Generate address
        SET app_address = CONCAT(
            FLOOR(100 + RAND() * 900), ' ',
            ELT(1 + FLOOR(RAND() * 10), 'Main Street', 'Commercial Avenue', 'Industrial Road', 'Residential Lane', 'Business Boulevard', 'Market Street', 'Park Avenue', 'Garden Road', 'Highway', 'Village Road'),
            ', ',
            ELT(1 + FLOOR(RAND() * 18), 'Barangay 1', 'Barangay 2', 'Barangay 3', 'Barangay 4', 'Barangay 5', 'Barangay 6', 'Barangay 7', 'Barangay 8', 'Barangay 9', 'Barangay 10', 'Downtown', 'Suburban', 'Rural', 'Urban Core', 'Industrial Zone', 'Commercial District', 'Residential Area', 'Mixed Zone'),
            ', City'
        );
        
        -- Project type (weighted: residential 35%, commercial 20%, industrial 15%, etc.)
        SET proj_type = CASE
            WHEN RAND() < 0.35 THEN 'residential'
            WHEN RAND() < 0.55 THEN 'commercial'
            WHEN RAND() < 0.70 THEN 'industrial'
            WHEN RAND() < 0.80 THEN 'agricultural'
            WHEN RAND() < 0.88 THEN 'mixed-use'
            WHEN RAND() < 0.93 THEN 'institutional'
            WHEN RAND() < 0.97 THEN 'recreational'
            ELSE 'residential-commercial'
        END;
        
        -- Project nature
        SET proj_nature = ELT(1 + FLOOR(RAND() * 3), 'new_development', 'improvement', 'others_renovation');
        
        -- Project location
        SET proj_location = ELT(1 + FLOOR(RAND() * 18), 'Barangay 1', 'Barangay 2', 'Barangay 3', 'Barangay 4', 'Barangay 5', 'Barangay 6', 'Barangay 7', 'Barangay 8', 'Barangay 9', 'Barangay 10', 'Downtown', 'Suburban', 'Rural', 'Urban Core', 'Industrial Zone', 'Commercial District', 'Residential Area', 'Mixed Zone');
        
        -- Generate area based on project type
        SET lot_val = CASE proj_type
            WHEN 'residential' THEN 80 + FLOOR(RAND() * 721)
            WHEN 'commercial' THEN 100 + FLOOR(RAND() * 1901)
            WHEN 'industrial' THEN 500 + FLOOR(RAND() * 4501)
            WHEN 'agricultural' THEN 1000 + FLOOR(RAND() * 9001)
            WHEN 'mixed-use' THEN 200 + FLOOR(RAND() * 2801)
            WHEN 'institutional' THEN 300 + FLOOR(RAND() * 1701)
            WHEN 'recreational' THEN 500 + FLOOR(RAND() * 4501)
            ELSE 150 + FLOOR(RAND() * 1351)
        END;
        
        -- Project area (30-90% of lot area depending on type)
        SET area_val = CASE proj_type
            WHEN 'agricultural' THEN lot_val * (0.5 + RAND() * 0.5)
            WHEN 'commercial' THEN lot_val * (0.4 + RAND() * 0.4)
            WHEN 'industrial' THEN lot_val * (0.4 + RAND() * 0.4)
            ELSE lot_val * (0.3 + RAND() * 0.4)
        END;
        
        SET building_val = area_val;
        
        -- Land right
        SET land_right_val = CASE
            WHEN RAND() < 0.80 THEN 'owner'
            WHEN RAND() < 0.95 THEN 'lessee'
            ELSE 'others'
        END;
        
        -- Project tenure
        SET tenure_val = CASE
            WHEN RAND() < 0.90 THEN 'permanent'
            ELSE 'temporary'
        END;
        
        -- Calculate cost based on project type and area
        SET cost_fig = CASE proj_type
            WHEN 'residential' THEN lot_val * (8000 + RAND() * 27000)
            WHEN 'commercial' THEN lot_val * (20000 + RAND() * 40000)
            WHEN 'industrial' THEN lot_val * (15000 + RAND() * 30000)
            WHEN 'agricultural' THEN lot_val * (3000 + RAND() * 9000)
            WHEN 'mixed-use' THEN lot_val * (18000 + RAND() * 32000)
            WHEN 'institutional' THEN lot_val * (25000 + RAND() * 30000)
            WHEN 'recreational' THEN lot_val * (12000 + RAND() * 28000)
            ELSE lot_val * (15000 + RAND() * 30000)
        END;
        
        -- Apply location premium/discount
        IF proj_location IN ('Downtown', 'Urban Core') THEN
            SET cost_fig = cost_fig * (1.15 + RAND() * 0.20);
        ELSEIF proj_location = 'Rural' THEN
            SET cost_fig = cost_fig * (0.75 + RAND() * 0.15);
        END IF;
        
        -- Apply area-based pricing
        IF lot_val > 2000 THEN
            SET cost_fig = cost_fig * (0.80 + RAND() * 0.15);
        ELSEIF lot_val < 150 THEN
            SET cost_fig = cost_fig * (1.05 + RAND() * 0.15);
        END IF;
        
        -- Ensure minimum cost
        IF cost_fig < 100000 THEN
            SET cost_fig = 100000 + RAND() * 400000;
        END IF;
        
        -- Generate cost in words (simplified)
        SET cost_words = CONCAT(
            CASE FLOOR(cost_fig / 1000000)
                WHEN 0 THEN ''
                WHEN 1 THEN 'One Million '
                WHEN 2 THEN 'Two Million '
                WHEN 3 THEN 'Three Million '
                WHEN 4 THEN 'Four Million '
                WHEN 5 THEN 'Five Million '
                WHEN 6 THEN 'Six Million '
                WHEN 7 THEN 'Seven Million '
                WHEN 8 THEN 'Eight Million '
                WHEN 9 THEN 'Nine Million '
                WHEN 10 THEN 'Ten Million '
                WHEN 11 THEN 'Eleven Million '
                WHEN 12 THEN 'Twelve Million '
                WHEN 13 THEN 'Thirteen Million '
                WHEN 14 THEN 'Fourteen Million '
                WHEN 15 THEN 'Fifteen Million '
                WHEN 16 THEN 'Sixteen Million '
                WHEN 17 THEN 'Seventeen Million '
                WHEN 18 THEN 'Eighteen Million '
                WHEN 19 THEN 'Nineteen Million '
                WHEN 20 THEN 'Twenty Million '
                ELSE CONCAT(FLOOR(cost_fig / 1000000), ' Million ')
            END,
            'Pesos'
        );
        
        -- Written notice subject
        SET notice_subject = CASE
            WHEN RAND() < 0.70 THEN 'yes'
            ELSE 'no'
        END;
        
        -- Release mode
        SET release_mode = CASE
            WHEN RAND() < 0.60 THEN 'pick_up'
            ELSE 'mail'
        END;
        
        -- Release address to
        SET release_to = CASE
            WHEN RAND() < 0.80 THEN 'applicant'
            ELSE 'authorized_representative'
        END;
        
        -- Generate coordinates (Philippines area)
        SET longitude_val = CONCAT('120.', LPAD(FLOOR(9800 + RAND() * 200), 4, '0'));
        SET latitude_val = CONCAT('14.', LPAD(FLOOR(6000 + RAND() * 100), 4, '0'));
        
        -- Generate dates (within last 3 years)
        SET created_date = DATE_SUB(NOW(), INTERVAL FLOOR(RAND() * 1095) DAY);
        SET updated_date = DATE_ADD(created_date, INTERVAL (5 + FLOOR(RAND() * 56)) DAY);
        
        -- Status
        SET status_val = ELT(1 + FLOOR(RAND() * 3), 'approved', 'pending', 'rejected');
        
        -- Insert record
        INSERT INTO application_forms (
            firstname,
            lastname,
            name_of_representative,
            applicant_address,
            representative_address,
            project_location,
            location_type,
            project_type,
            project_area,
            project_lifespan,
            project_significance,
            project_classification,
            site_zoning,
            land_uses,
            applicant_name,
            representative_name,
            project_area_sqm,
            project_nature,
            right_over_land,
            project_tenure,
            lot_area,
            building_improvement_area,
            existing_land_use_site,
            project_cost_numeric,
            project_cost_words,
            is_subject_of_notice,
            notice_offices,
            preferred_mode_of_release,
            address_to,
            existing_land_uses,
            written_notices,
            other_hsrc_offices,
            dates_filed,
            actions_taken,
            applicant_signature,
            representative_signature,
            longitude,
            latitude,
            status,
            created_at,
            updated_at
        ) VALUES (
            first_name,
            last_name,
            CASE WHEN RAND() < 0.30 THEN CONCAT('Representative ', app_name) ELSE NULL END,
            app_address,
            CASE WHEN RAND() < 0.30 THEN app_address ELSE NULL END,
            proj_location,
            1,
            proj_type,
            ROUND(area_val, 2),
            tenure_val,
            'local',
            CONCAT(proj_type, ' development'),
            CASE proj_type
                WHEN 'residential' THEN 'R-1'
                WHEN 'commercial' THEN 'C-2'
                WHEN 'industrial' THEN 'I-1'
                WHEN 'agricultural' THEN 'A-1'
                ELSE 'M-1'
            END,
            CONCAT(proj_type, ' development'),
            app_name,
            CASE WHEN RAND() < 0.30 THEN CONCAT('Representative ', app_name) ELSE NULL END,
            ROUND(area_val, 2),
            proj_nature,
            land_right_val,
            tenure_val,
            ROUND(lot_val, 2),
            ROUND(building_val, 2),
            CONCAT(proj_type, ' development'),
            ROUND(cost_fig, 2),
            cost_words,
            notice_subject,
            CASE WHEN notice_subject = 'yes' THEN 'City Planning Office, Building Official' ELSE NULL END,
            release_mode,
            release_to,
            CONCAT(proj_type, ' development'),
            notice_subject,
            CASE WHEN notice_subject = 'yes' THEN 'City Planning Office, Building Official' ELSE NULL END,
            DATE_FORMAT(created_date, '%Y-%m-%d'),
            CASE status_val
                WHEN 'approved' THEN 'Application approved, Site inspection completed'
                WHEN 'pending' THEN 'Application under review, Initial review completed'
                ELSE 'Application requires additional documentation'
            END,
            app_name,
            CASE WHEN RAND() < 0.30 THEN CONCAT('Representative ', app_name) ELSE NULL END,
            longitude_val,
            latitude_val,
            status_val,
            created_date,
            updated_date
        );
        
        SET i = i + 1;
    END WHILE;
END$$

DELIMITER ;

-- =====================================================
-- EXECUTE THE PROCEDURE
-- =====================================================
-- Change the number to insert more/fewer records
CALL InsertMockApplicationForms(1000);

-- Drop the procedure after use (optional)
DROP PROCEDURE IF EXISTS InsertMockApplicationForms;

-- Re-enable foreign key checks
SET FOREIGN_KEY_CHECKS = 1;

-- =====================================================
-- VERIFY DATA
-- =====================================================
SELECT 
    COUNT(*) as total_records,
    MIN(created_at) as earliest_date,
    MAX(created_at) as latest_date,
    AVG(project_cost_numeric) as avg_cost,
    SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) as approved_count,
    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_count,
    SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected_count
FROM application_forms
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 3 YEAR);

-- Show sample records
SELECT 
    applicant_name,
    project_type,
    project_location,
    lot_area,
    project_cost_numeric,
    status,
    created_at
FROM application_forms
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 3 YEAR)
ORDER BY created_at DESC
LIMIT 10;
