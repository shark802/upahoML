# Docker Deployment - Complete Setup

## 🐳 What's Included

✅ **Dockerfile** - Containerizes the application
✅ **docker-compose.yml** - Easy local development
✅ **docker-compose.prod.yml** - Production configuration
✅ **.env.example** - Environment variable template
✅ **.dockerignore** - Excludes unnecessary files

## 🚀 Quick Start

### 1. Create Environment File

```bash
# Copy example file
cp .env.example .env

# Edit .env with your database credentials
# DB_HOST=srv1322.hstgr.io
# DB_USER=u520834156_uPAHOZone25
# DB_PASSWORD=Y+;a+*1y
# DB_NAME=u520834156_dbUPAHOZoning
# DB_PORT=3306
```

### 2. Build and Run

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f
```

### 3. Test

```bash
# Check status
curl http://localhost:5000/api/check

# Train models
curl http://localhost:5000/api/train

# Make prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"prediction_type":"land_cost_future","target_years":10,"data":{"lot_area":200,"project_area":150,"project_type":"residential","location":"Downtown"}}'
```

## 📦 What Docker Sets Up Automatically

- ✅ Python 3.11 environment
- ✅ All Python dependencies installed
- ✅ MySQL client libraries
- ✅ Database connection configured
- ✅ Models directory created
- ✅ Gunicorn web server
- ✅ Health checks enabled
- ✅ Auto-restart on failure

## 🔧 Configuration

All settings in `.env` file:

```env
DB_HOST=srv1322.hstgr.io
DB_USER=u520834156_uPAHOZone25
DB_PASSWORD=Y+;a+*1y
DB_NAME=u520834156_dbUPAHOZoning
DB_PORT=3306
```

## 📋 Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Restart
docker-compose restart

# View logs
docker-compose logs -f

# Rebuild
docker-compose up -d --build

# Execute commands
docker-compose exec web bash
```

## 🌐 Deploy to Heroku with Docker

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

## ✅ Benefits

- **Consistent**: Same environment everywhere
- **Easy**: One command to deploy
- **Isolated**: No system conflicts
- **Portable**: Run anywhere Docker runs
- **Scalable**: Easy to scale horizontally

## 📚 More Info

See `DOCKER_DEPLOYMENT_GUIDE.md` for detailed instructions.

