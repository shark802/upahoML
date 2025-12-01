<?php
/**
 * Simplified version for troubleshooting
 * Use this if the main file doesn't work
 */

// Turn on error display temporarily
ini_set('display_errors', 1);
error_reporting(E_ALL);

header('Content-Type: application/json');

// Simple test first
if (!function_exists('curl_init')) {
    die(json_encode(['success' => false, 'error' => 'cURL not available']));
}

$input = file_get_contents('php://input');
$data = json_decode($input, true);

if (!$data) {
    die(json_encode(['success' => false, 'error' => 'Invalid JSON']));
}

$api_url = 'https://upaho-883f1ffc88a8.herokuapp.com';
$endpoint = $api_url . '/predict/land_cost_future';

// Simple request
$ch = curl_init($endpoint);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
curl_setopt($ch, CURLOPT_TIMEOUT, 30);
curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, false); // Disable SSL verification for testing
curl_setopt($ch, CURLOPT_SSL_VERIFYHOST, false);

$response = curl_exec($ch);
$http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$error = curl_error($ch);
curl_close($ch);

if ($response === false) {
    die(json_encode(['success' => false, 'error' => 'cURL error: ' . $error]));
}

if ($http_code !== 200) {
    die(json_encode(['success' => false, 'error' => "HTTP $http_code: " . substr($response, 0, 200)]));
}

echo $response;
?>


