# Docker Quick Start Guide

## 🚀 Quick Setup (3 Steps)

### Step 1: Create Environment File

```bash
# Create .env file with your database credentials
cat > .env << EOF
DB_HOST=srv1322.hstgr.io
DB_USER=u520834156_uPAHOZone25
DB_PASSWORD=Y+;a+*1y
DB_NAME=u520834156_dbUPAHOZoning
DB_PORT=3306
EOF
```

Or manually create `.env` file:
```env
DB_HOST=srv1322.hstgr.io
DB_USER=u520834156_uPAHOZone25
DB_PASSWORD=Y+;a+*1y
DB_NAME=u520834156_dbUPAHOZoning
DB_PORT=3306
```

### Step 2: Build and Start

```bash
# Build and start container
docker-compose up -d --build

# View logs
docker-compose logs -f
```

### Step 3: Test

```bash
# Check connection
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

## 📋 Common Commands

```bash
# Start containers
docker-compose up -d

# Stop containers
docker-compose down

# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Rebuild after code changes
docker-compose up -d --build

# Check status
docker-compose ps

# Execute command in container
docker-compose exec web bash
```

## 🔧 Configuration

All database settings are in `.env` file or environment variables:

- `DB_HOST` - Database hostname
- `DB_USER` - Database username
- `DB_PASSWORD` - Database password
- `DB_NAME` - Database name
- `DB_PORT` - Database port (default: 3306)

## 🌐 Access Points

- **API**: http://localhost:5000
- **Health Check**: http://localhost:5000/api/check
- **Train Models**: http://localhost:5000/api/train
- **Predict**: http://localhost:5000/api/predict

## 🐳 Deploy to Production

### Heroku with Docker

```bash
# Set Heroku to use Docker
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

### Other Platforms

See `DOCKER_DEPLOYMENT_GUIDE.md` for AWS, Google Cloud, DigitalOcean, etc.

## ✅ That's It!

Your API is now running in Docker with:
- ✅ All dependencies installed
- ✅ Database configured
- ✅ Models directory ready
- ✅ Health checks enabled
- ✅ Auto-restart on failure

