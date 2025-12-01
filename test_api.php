<?php
/**
 * Test file to diagnose API connection issues
 * Access this file directly in your browser to test the API connection
 */

header('Content-Type: text/html; charset=utf-8');
?>
<!DOCTYPE html>
<html>
<head>
    <title>API Connection Test</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .success { color: green; background: #e8f5e9; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .error { color: red; background: #ffebee; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .info { color: blue; background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; }
        pre { background: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>🔍 API Connection Diagnostic Test</h1>
    
    <?php
    echo "<div class='info'><strong>PHP Version:</strong> " . phpversion() . "</div>";
    
    // Test 1: Check cURL
    echo "<h2>Test 1: cURL Extension</h2>";
    if (function_exists('curl_init')) {
        echo "<div class='success'>✅ cURL is enabled</div>";
        $curl_version = curl_version();
        echo "<div class='info'>cURL Version: " . $curl_version['version'] . "</div>";
        echo "<div class='info'>SSL Version: " . ($curl_version['ssl_version'] ?? 'Not available') . "</div>";
    } else {
        echo "<div class='error'>❌ cURL is NOT enabled. Please enable the cURL extension in PHP.</div>";
        exit;
    }
    
    // Test 2: Check JSON
    echo "<h2>Test 2: JSON Extension</h2>";
    if (function_exists('json_encode')) {
        echo "<div class='success'>✅ JSON is enabled</div>";
    } else {
        echo "<div class='error'>❌ JSON is NOT enabled. Please enable the JSON extension in PHP.</div>";
        exit;
    }
    
    // Test 3: Test API Connection
    echo "<h2>Test 3: API Connection Test</h2>";
    
    $api_url = 'https://upaho-883f1ffc88a8.herokuapp.com';
    $endpoint = $api_url . '/predict/land_cost_future';
    
    echo "<div class='info'><strong>Testing endpoint:</strong> $endpoint</div>";
    
    // Test data
    $test_data = [
        'target_years' => 10,
        'data' => [
            'lot_area' => 200,
            'project_area' => 150,
            'project_type' => 'residential',
            'location' => 'Downtown',
            'year' => 2024,
            'month' => 1,
            'age' => 35
        ]
    ];
    
    $ch = curl_init($endpoint);
    curl_setopt_array($ch, [
        CURLOPT_POST => true,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_HTTPHEADER => [
            'Content-Type: application/json',
        ],
        CURLOPT_POSTFIELDS => json_encode($test_data),
        CURLOPT_TIMEOUT => 30,
        CURLOPT_CONNECTTIMEOUT => 10,
        CURLOPT_SSL_VERIFYPEER => true,
        CURLOPT_SSL_VERIFYHOST => 2,
    ]);
    
    $response = curl_exec($ch);
    $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $curl_error = curl_error($ch);
    $curl_errno = curl_errno($ch);
    curl_close($ch);
    
    if ($response === false || !empty($curl_error)) {
        echo "<div class='error'>❌ Connection failed!</div>";
        echo "<div class='error'><strong>Error:</strong> $curl_error</div>";
        echo "<div class='error'><strong>Error Code:</strong> $curl_errno</div>";
        
        if ($curl_errno == CURLE_SSL_CONNECT_ERROR || $curl_errno == 60) {
            echo "<div class='info'><strong>SSL Certificate Issue Detected</strong><br>";
            echo "If you're getting SSL certificate errors, you may need to update your server's CA certificates or temporarily disable SSL verification (NOT recommended for production).</div>";
        }
    } else {
        echo "<div class='success'>✅ Connection successful!</div>";
        echo "<div class='info'><strong>HTTP Status Code:</strong> $http_code</div>";
        
        if ($http_code == 200) {
            $result = json_decode($response, true);
            if ($result) {
                echo "<div class='success'>✅ Valid JSON response received</div>";
                echo "<h3>Response Preview:</h3>";
                echo "<pre>" . json_encode($result, JSON_PRETTY_PRINT) . "</pre>";
            } else {
                echo "<div class='error'>⚠️ Response received but not valid JSON</div>";
                echo "<pre>" . htmlspecialchars(substr($response, 0, 1000)) . "</pre>";
            }
        } else {
            echo "<div class='error'>⚠️ API returned HTTP $http_code</div>";
            echo "<pre>" . htmlspecialchars(substr($response, 0, 1000)) . "</pre>";
        }
    }
    
    // Test 4: Test PHP file
    echo "<h2>Test 4: PHP Backend File Test</h2>";
    $php_file = __DIR__ . '/predict_land_cost_api.php';
    if (file_exists($php_file)) {
        echo "<div class='success'>✅ predict_land_cost_api.php exists</div>";
        
        // Test if file is readable
        if (is_readable($php_file)) {
            echo "<div class='success'>✅ File is readable</div>";
        } else {
            echo "<div class='error'>❌ File is not readable. Check file permissions.</div>";
        }
        
        // Check for syntax errors
        $output = [];
        $return_var = 0;
        exec("php -l " . escapeshellarg($php_file) . " 2>&1", $output, $return_var);
        if ($return_var === 0) {
            echo "<div class='success'>✅ No PHP syntax errors</div>";
        } else {
            echo "<div class='error'>❌ PHP syntax errors found:</div>";
            echo "<pre>" . htmlspecialchars(implode("\n", $output)) . "</pre>";
        }
    } else {
        echo "<div class='error'>❌ predict_land_cost_api.php not found at: $php_file</div>";
    }
    
    // Test 5: Server Information
    echo "<h2>Test 5: Server Information</h2>";
    echo "<div class='info'>";
    echo "<strong>Server Software:</strong> " . ($_SERVER['SERVER_SOFTWARE'] ?? 'Unknown') . "<br>";
    echo "<strong>Document Root:</strong> " . ($_SERVER['DOCUMENT_ROOT'] ?? 'Unknown') . "<br>";
    echo "<strong>Script Path:</strong> " . __FILE__ . "<br>";
    echo "<strong>Current Directory:</strong> " . __DIR__ . "<br>";
    echo "<strong>Allow URL Fopen:</strong> " . (ini_get('allow_url_fopen') ? 'Yes' : 'No') . "<br>";
    echo "</div>";
    ?>
    
    <h2>Next Steps</h2>
    <div class='info'>
        <ol>
            <li>If all tests pass, try accessing <code>land_cost_prediction_ui.html</code> in your browser</li>
            <li>If you see errors, check your server's error logs</li>
            <li>Make sure both files are uploaded to the same directory</li>
            <li>Check file permissions (should be 644 for PHP files)</li>
        </ol>
    </div>
</body>
</html>


