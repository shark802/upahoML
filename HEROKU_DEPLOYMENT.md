# Heroku Deployment Guide

## Prerequisites

1. Heroku account (sign up at https://heroku.com)
2. Heroku CLI installed (https://devcenter.heroku.com/articles/heroku-cli)
3. Git repository initialized

## Quick Deployment Steps

### 1. Login to Heroku
```bash
heroku login
```

### 2. Create Heroku App
```bash
heroku create your-app-name
```

### 3. Set Environment Variables (Database Config)
```bash
heroku config:set DB_HOST=your-db-host
heroku config:set DB_USER=your-db-user
heroku config:set DB_PASSWORD=your-db-password
heroku config:set DB_NAME=your-database-name
```

Or if you have a `config.json` file, you can upload it (but environment variables are more secure).

### 4. Deploy to Heroku
```bash
git add .
git commit -m "Add Heroku deployment files"
git push heroku main
```

### 5. Check Deployment Status
```bash
heroku logs --tail
```

## API Endpoints

Once deployed, your API will be available at: `https://your-app-name.herokuapp.com`

### Available Endpoints:

1. **Health Check**: `GET /`
2. **Land Cost Prediction**: `POST /predict/land_cost`
3. **Future Land Cost Prediction**: `POST /predict/land_cost_future`
4. **Approval Probability**: `POST /predict/approval`
5. **Processing Time**: `POST /predict/processing_time`

### Example Request:

```bash
curl -X POST https://your-app-name.herokuapp.com/predict/land_cost_future \
  -H "Content-Type: application/json" \
  -d '{
    "target_years": 10,
    "data": {
      "lot_area": 200,
      "project_area": 150,
      "project_type": "residential",
      "location": "Downtown",
      "age": 35
    }
  }'
```

## Important Notes

1. **Models**: You need to train models before deployment. Models should be in the `models/` directory or uploaded separately.

2. **Database**: Make sure your database is accessible from Heroku (use a cloud database like Heroku Postgres, AWS RDS, or similar).

3. **File Size**: Heroku has a 500MB slug size limit. Large model files might need to be stored externally (S3, etc.).

4. **Dyno Types**: The free tier has limitations. Consider upgrading for production use.

## Troubleshooting

### Build Fails
- Check `requirements.txt` is in root directory
- Verify Python version in `runtime.txt`
- Check Heroku logs: `heroku logs --tail`

### Models Not Found
- Train models locally first
- Upload models to external storage (S3) and download on startup
- Or include models in Git (if small enough)

### Database Connection Issues
- Verify environment variables are set correctly
- Check database allows connections from Heroku IPs
- Use Heroku Postgres addon for easier setup

## Alternative: Deploy Models Separately

If model files are too large, you can:
1. Store models in AWS S3 or similar
2. Download models on app startup
3. Or use Heroku's ephemeral filesystem (models will be lost on restart)

