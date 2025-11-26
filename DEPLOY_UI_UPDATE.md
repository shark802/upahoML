# UI Deployment Update

## Changes Made

1. **Updated `app.py`**:
   - Root route (`/`) now serves the HTML UI
   - Added `/api` route for API information
   - Imported `send_from_directory` to serve static files

2. **Updated `land_cost_prediction_ui.html`**:
   - Changed API endpoint from `predict_land_cost.php` to `/predict/land_cost_future`
   - Updated fetch call to use Flask API endpoint
   - Removed dependency on PHP

3. **Updated `.slugignore`**:
   - Added exception for `land_cost_prediction_ui.html` so it's included in deployment

## Deployment Steps

```bash
# 1. Commit the changes
git add app.py land_cost_prediction_ui.html .slugignore
git commit -m "Add UI support - serve HTML and update API endpoints"

# 2. Deploy to Heroku
git push heroku main

# 3. Test the UI
heroku open
```

## Access Points

- **UI**: `https://your-app-name.herokuapp.com/` - Shows the prediction form
- **API Info**: `https://your-app-name.herokuapp.com/api` - Shows API documentation
- **API Endpoint**: `https://your-app-name.herokuapp.com/predict/land_cost_future` - Prediction endpoint

## Testing

1. Visit your Heroku app URL
2. You should see the Land Cost Prediction form
3. Fill in the form and click "Predict Land Cost"
4. Results should display below

## Troubleshooting

### UI not showing
- Check that `land_cost_prediction_ui.html` is in the root directory
- Verify `.slugignore` includes the HTML file
- Check Heroku logs: `heroku logs --tail`

### API errors
- Check that models are trained and available
- Verify database connection
- Check logs for specific error messages

