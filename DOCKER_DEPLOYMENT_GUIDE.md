# Docker Deployment Guide

## Overview

This guide shows how to deploy the UPAHO ML Prediction API using Docker. Docker containerizes the application with all dependencies, making deployment consistent across environments.

## Prerequisites

- Docker installed ([Download Docker](https://www.docker.com/products/docker-desktop))
- Docker Compose installed (included with Docker Desktop)
- Database credentials ready

## Quick Start

### 1. Build and Run with Docker Compose

```bash
# Copy environment file
cp .env.example .env

# Edit .env file with your database credentials
# (or set environment variables)

# Build and start containers
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 2. Test the API

```bash
# Check health
curl http://localhost:5000/api/check

# Train models
curl http://localhost:5000/api/train

# Test prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_type": "land_cost_future",
    "target_years": 10,
    "data": {
      "lot_area": 200,
      "project_area": 150,
      "project_type": "residential",
      "location": "Downtown"
    }
  }'
```

## Environment Variables

### Option 1: Using .env File

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your credentials:
   ```env
   DB_HOST=srv1322.hstgr.io
   DB_USER=u520834156_uPAHOZone25
   DB_PASSWORD=Y+;a+*1y
   DB_NAME=u520834156_dbUPAHOZoning
   DB_PORT=3306
   ```

3. Docker Compose will automatically load `.env` file

### Option 2: Using Environment Variables

```bash
export DB_HOST=srv1322.hstgr.io
export DB_USER=u520834156_uPAHOZone25
export DB_PASSWORD=Y+;a+*1y
export DB_NAME=u520834156_dbUPAHOZoning
export DB_PORT=3306

docker-compose up -d
```

### Option 3: Inline with Docker Compose

```bash
DB_HOST=srv1322.hstgr.io DB_USER=u520834156_uPAHOZone25 DB_PASSWORD=Y+;a+*1y DB_NAME=u520834156_dbUPAHOZoning docker-compose up -d
```

## Docker Commands

### Build Image
```bash
docker build -t upaho-ml-api .
```

### Run Container
```bash
docker run -d \
  -p 5000:5000 \
  -e DB_HOST=srv1322.hstgr.io \
  -e DB_USER=u520834156_uPAHOZone25 \
  -e DB_PASSWORD=Y+;a+*1y \
  -e DB_NAME=u520834156_dbUPAHOZoning \
  -e DB_PORT=3306 \
  -v $(pwd)/models:/app/models \
  --name upaho-ml-api \
  upaho-ml-api
```

### View Logs
```bash
# Docker Compose
docker-compose logs -f

# Docker
docker logs -f upaho-ml-api
```

### Stop Container
```bash
# Docker Compose
docker-compose down

# Docker
docker stop upaho-ml-api
docker rm upaho-ml-api
```

### Restart Container
```bash
# Docker Compose
docker-compose restart

# Docker
docker restart upaho-ml-api
```

### Execute Commands in Container
```bash
# Docker Compose
docker-compose exec web bash

# Docker
docker exec -it upaho-ml-api bash
```

## Production Deployment

### Using Docker Compose (Production)

```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up -d

# With environment variables
DB_HOST=srv1322.hstgr.io \
DB_USER=u520834156_uPAHOZone25 \
DB_PASSWORD=Y+;a+*1y \
DB_NAME=u520834156_dbUPAHOZoning \
docker-compose -f docker-compose.prod.yml up -d
```

### Deploy to Cloud Platforms

#### Option 1: Heroku with Docker

```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Set stack to container
heroku stack:set container

# Set environment variables
heroku config:set DB_HOST=srv1322.hstgr.io
heroku config:set DB_USER=u520834156_uPAHOZone25
heroku config:set DB_PASSWORD=Y+;a+*1y
heroku config:set DB_NAME=u520834156_dbUPAHOZoning
heroku config:set DB_PORT=3306

# Deploy
git push heroku main
```

#### Option 2: AWS ECS / EC2

1. Build and push to ECR:
   ```bash
   # Build
   docker build -t upaho-ml-api .

   # Tag
   docker tag upaho-ml-api:latest your-account.dkr.ecr.region.amazonaws.com/upaho-ml-api:latest

   # Push
   docker push your-account.dkr.ecr.region.amazonaws.com/upaho-ml-api:latest
   ```

2. Deploy to ECS with environment variables set

#### Option 3: Google Cloud Run

```bash
# Build
gcloud builds submit --tag gcr.io/your-project/upaho-ml-api

# Deploy
gcloud run deploy upaho-ml-api \
  --image gcr.io/your-project/upaho-ml-api \
  --platform managed \
  --region us-central1 \
  --set-env-vars DB_HOST=srv1322.hstgr.io,DB_USER=u520834156_uPAHOZone25,DB_PASSWORD=Y+;a+*1y,DB_NAME=u520834156_dbUPAHOZoning
```

#### Option 4: DigitalOcean App Platform

1. Connect GitHub repository
2. Select Dockerfile
3. Set environment variables in dashboard
4. Deploy

## Persistent Storage

Models are stored in `/app/models` inside the container. To persist models:

### Docker Compose (Automatic)
The `docker-compose.yml` mounts `./models` directory, so models persist on your host.

### Docker Run (Manual)
```bash
docker run -v $(pwd)/models:/app/models ...
```

### Production
Use Docker volumes:
```bash
docker volume create models-data
docker run -v models-data:/app/models ...
```

## Health Checks

The container includes health checks. Monitor health:

```bash
# Check container health
docker ps

# Health check endpoint
curl http://localhost:5000/api/check
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs

# Check container status
docker-compose ps
```

### Database connection fails
```bash
# Test connection from container
docker-compose exec web python -c "
import mysql.connector
import os
conn = mysql.connector.connect(
    host=os.environ.get('DB_HOST'),
    user=os.environ.get('DB_USER'),
    password=os.environ.get('DB_PASSWORD'),
    database=os.environ.get('DB_NAME')
)
print('Connected!')
"
```

### Models not persisting
```bash
# Check volume mount
docker-compose exec web ls -la /app/models

# Check host directory
ls -la ./models
```

### Rebuild after code changes
```bash
# Rebuild and restart
docker-compose up -d --build
```

## Development Workflow

### Local Development
```bash
# Start services
docker-compose up

# Make code changes
# (files are mounted, changes reflect immediately)

# Restart if needed
docker-compose restart
```

### Testing
```bash
# Run tests in container
docker-compose exec web python -m pytest

# Or run locally with same environment
docker-compose run --rm web python your_test.py
```

## Complete Setup Example

```bash
# 1. Clone/navigate to project
cd c:\xampp-25\htdocs\upaho_ml\upahoML

# 2. Create .env file
cat > .env << EOF
DB_HOST=srv1322.hstgr.io
DB_USER=u520834156_uPAHOZone25
DB_PASSWORD=Y+;a+*1y
DB_NAME=u520834156_dbUPAHOZoning
DB_PORT=3306
EOF

# 3. Build and start
docker-compose up -d --build

# 4. Check logs
docker-compose logs -f

# 5. Test connection
curl http://localhost:5000/api/check

# 6. Train models
curl http://localhost:5000/api/train

# 7. Test prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"prediction_type":"land_cost_future","target_years":10,"data":{"lot_area":200,"project_area":150,"project_type":"residential","location":"Downtown"}}'
```

## Benefits of Docker

✅ **Consistent Environment**: Same setup everywhere
✅ **Easy Deployment**: One command to deploy
✅ **Isolation**: No conflicts with system packages
✅ **Scalability**: Easy to scale horizontally
✅ **Reproducibility**: Same results every time
✅ **Portability**: Run anywhere Docker runs

## Next Steps

1. ✅ Build Docker image
2. ✅ Set environment variables
3. ✅ Start container
4. ✅ Test connection
5. ✅ Train models
6. ✅ Deploy to production platform

