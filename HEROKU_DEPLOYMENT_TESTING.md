# Deploy Testing Page to Heroku

## ✅ Testing Page Added to Heroku

The testing page is now integrated into your Heroku Flask app!

---

## 🚀 How to Access on Heroku

### Main Testing Page:
```
https://upaho-883f1ffc88a8.herokuapp.com/test
```

### Original UI (Still Available):
```
https://upaho-883f1ffc88a8.herokuapp.com/
```

---

## 📋 What Was Changed

### 1. Added Route to `app.py`
```python
@app.route('/test')
def test_page():
    """Serve the testing page for predictions"""
    return send_from_directory(os.path.dirname(__file__), 'test_predictions.html')
```

### 2. Updated `test_predictions.html`
- Automatically detects if running on Heroku
- Uses direct API endpoint on Heroku
- Uses PHP backend when running locally

---

## 🔄 How It Works

### On Heroku:
```
User → /test → Flask serves HTML → JavaScript → /predict/land_cost_future → Model → Results
```

### Locally:
```
User → test_predictions.html → JavaScript → predict_land_cost_api.php → Heroku API → Results
```

---

## 📤 Deployment Steps

### Step 1: Commit Files
```bash
git add test_predictions.html app.py
git commit -m "Add testing page for predictions"
```

### Step 2: Push to Heroku
```bash
git push heroku main
```

### Step 3: Verify Deployment
```bash
heroku logs --tail
```

### Step 4: Test the Page
Open in browser:
```
https://upaho-883f1ffc88a8.herokuapp.com/test
```

---

## 🧪 Testing on Heroku

### Quick Test:
1. Open: `https://upaho-883f1ffc88a8.herokuapp.com/test`
2. Click: "Fill Sample Data"
3. Click: "🔮 Predict Land Property Value"
4. View: Results

### What to Expect:
- ✅ Page loads correctly
- ✅ Form submits successfully
- ✅ Predictions display
- ✅ All features working
- ✅ No CORS errors (same origin)

---

## 🔧 API Endpoints Used

### On Heroku:
- **Testing Page:** `/test` (serves HTML)
- **Prediction API:** `/predict/land_cost_future` (POST)
- **Alternative:** `/api/predict` (POST)

### Direct API Testing:
You can also test the API directly:
```bash
curl -X POST https://upaho-883f1ffc88a8.herokuapp.com/predict/land_cost_future \
  -H "Content-Type: application/json" \
  -d '{
    "target_years": 10,
    "data": {
      "lot_area": 200,
      "project_area": 150,
      "project_type": "residential",
      "location": "Downtown",
      "latitude": 14.5995,
      "longitude": 120.9842,
      "year": 2024,
      "month": 12,
      "age": 35
    }
  }'
```

---

## 📁 Files to Deploy

Make sure these files are in your Heroku app:
- ✅ `test_predictions.html` - Testing page
- ✅ `app.py` - Flask app (with /test route)
- ✅ `land_predictions.py` - Model code
- ✅ `models/` - Trained models (or auto-train on first use)

---

## 🎯 Deployment Checklist

Before deploying:
- [ ] `test_predictions.html` exists
- [ ] `app.py` has `/test` route
- [ ] Models are trained (or auto-train enabled)
- [ ] Database connection configured
- [ ] All dependencies in `requirements.txt`

After deploying:
- [ ] Test page loads: `/test`
- [ ] Form submission works
- [ ] Predictions display correctly
- [ ] No console errors (F12)
- [ ] API endpoints respond

---

## 🐛 Troubleshooting

### Issue: 404 on /test
**Solution:**
- Check `app.py` has the route
- Verify `test_predictions.html` is in root directory
- Check Heroku logs: `heroku logs --tail`

### Issue: API errors
**Solution:**
- Verify models are trained: `POST /api/train`
- Check database connection
- Review Heroku logs

### Issue: CORS errors
**Solution:**
- Shouldn't happen (same origin on Heroku)
- If using from different domain, check CORS settings

---

## 📊 Testing Scenarios

### Test 1: Basic Prediction
```
1. Open /test
2. Fill Sample Data
3. Predict
4. Verify results show
```

### Test 2: With Coordinates
```
1. Enter coordinates
2. Predict
3. Verify distance features work
```

### Test 3: Different Locations
```
1. Test Downtown (premium)
2. Test Rural (economy)
3. Compare results
```

---

## 🔗 URLs

**Heroku Testing Page:**
```
https://upaho-883f1ffc88a8.herokuapp.com/test
```

**Heroku Original UI:**
```
https://upaho-883f1ffc88a8.herokuapp.com/
```

**Heroku API:**
```
https://upaho-883f1ffc88a8.herokuapp.com/predict/land_cost_future
```

---

## ✅ Quick Deploy Command

```bash
# Add and commit
git add test_predictions.html app.py
git commit -m "Add testing page"

# Deploy to Heroku
git push heroku main

# Check logs
heroku logs --tail
```

---

**Status:** ✅ Ready to Deploy  
**Testing Page:** `/test` route added  
**Auto-Detection:** Works on Heroku and locally

