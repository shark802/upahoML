# How to Call/Use the Testing Page

## 🚀 Quick Access

### Option 1: Local (XAMPP)
Open in your browser:
```
http://localhost/upahoML/test_predictions.html
```

### Option 2: Online Server
Open in your browser:
```
https://upahozoning.bccbsis.com/test_predictions.html
```

### Option 3: Direct File Path
If running locally, you can also open directly:
```
file:///C:/xampp-25/htdocs/upaho_ml/upahoML/test_predictions.html
```

---

## 📋 Step-by-Step Usage

### Step 1: Open the Page
1. Start your web server (XAMPP Apache)
2. Open your browser
3. Navigate to: `http://localhost/upahoML/test_predictions.html`

### Step 2: Quick Test (Easiest)
1. Click the **"📋 Fill Sample Data"** button
   - This fills the form with realistic test data
2. Click **"🔮 Predict Land Property Value"** button
3. Wait for results (2-5 seconds)
4. View predictions on the right side

### Step 3: Custom Test
1. Fill in the form manually:
   - **Lot Area:** Enter size in square meters (e.g., 200)
   - **Project Area:** Enter building area (e.g., 150)
   - **Project Type:** Select from dropdown
   - **Location:** Select location
   - **Coordinates:** Optional but recommended
   - **Zoning:** Optional
2. Click **"🔮 Predict Land Property Value"**
3. View results

---

## 🎯 Quick Test Examples

### Example 1: Premium Location Test
```
1. Click "Fill Sample Data"
2. Click "Predict"
3. See Downtown premium pricing
```

### Example 2: Custom Location Test
```
1. Lot Area: 300
2. Project Area: 200
3. Project Type: Commercial
4. Location: Commercial District
5. Click "Use City Center" for coordinates
6. Click "Predict"
```

### Example 3: Rural Area Test
```
1. Lot Area: 500
2. Project Area: 300
3. Project Type: Agricultural
4. Location: Rural
5. Latitude: 14.6500
6. Longitude: 121.0500
7. Click "Predict"
```

---

## 🔧 What Happens When You Click "Predict"

1. **Form Data Collected**
   - All form fields are gathered
   - Optional fields included if filled

2. **API Call**
   - Data sent to: `predict_land_cost_api.php`
   - PHP forwards to Heroku API
   - API processes with all 23 features

3. **Results Displayed**
   - Current year prediction
   - Future prediction (5-10 years)
   - Three scenarios
   - Yearly breakdown
   - Features used

---

## 📱 Browser Requirements

**Supported Browsers:**
- ✅ Chrome (recommended)
- ✅ Firefox
- ✅ Edge
- ✅ Safari
- ✅ Opera

**Required:**
- JavaScript enabled
- Modern browser (last 2 years)

---

## 🐛 Troubleshooting

### Issue: Page doesn't load
**Solution:**
- Check XAMPP Apache is running
- Verify file exists: `test_predictions.html`
- Check URL is correct

### Issue: "Cannot connect to server"
**Solution:**
- Verify `predict_land_cost_api.php` is in same directory
- Check PHP is working: `http://localhost/upahoML/predict_land_cost_api.php`
- Check browser console (F12) for errors

### Issue: No results showing
**Solution:**
- Open browser console (F12)
- Check for JavaScript errors
- Verify API response in Network tab
- Check PHP error logs

### Issue: Predictions seem wrong
**Solution:**
- Verify model is trained (check `models/` folder)
- Check data quality
- Review feature values in form

---

## 🧪 Testing Checklist

Before testing, verify:
- [ ] XAMPP Apache is running
- [ ] `test_predictions.html` exists
- [ ] `predict_land_cost_api.php` exists in same folder
- [ ] Models are trained (check `models/` folder)
- [ ] Database connection is working

---

## 📊 What You'll See

### Successful Prediction Shows:
1. **Model Info:** R² score, features used
2. **Current Value:** Cost per sqm and total value
3. **Future Value:** 5-10 year projection
4. **Scenarios:** Optimistic, Realistic, Conservative
5. **Yearly Breakdown:** Year-by-year progression
6. **Features Used:** List of active features

### Example Output:
```
Current Year: 43,100 PHP/sqm
Total Value: 8,620,000 PHP

Future (10 years): 58,745 PHP/sqm
Total Value: 11,749,000 PHP

Appreciation: +36.3% over 10 years
Rate: +3.14% per year
```

---

## 🎓 Testing Tips

1. **Start Simple:**
   - Use "Fill Sample Data" first
   - See how it works
   - Then customize

2. **Test Variations:**
   - Try different locations
   - Test different sizes
   - Compare project types

3. **Compare Results:**
   - Premium vs Economy locations
   - Small vs Large lots
   - Residential vs Commercial

4. **Check Features:**
   - With coordinates vs without
   - With zoning vs without
   - See how features affect predictions

---

## 🔗 Related Files

- **Testing Page:** `test_predictions.html`
- **API Backend:** `predict_land_cost_api.php`
- **Model Code:** `land_predictions.py`
- **Guide:** `TESTING_PAGE_GUIDE.md`

---

## ✅ Quick Start Command

**Just open this URL in your browser:**
```
http://localhost/upahoML/test_predictions.html
```

Then click **"Fill Sample Data"** and **"Predict"**!

---

**That's it!** The testing page is ready to use. Just open it in your browser and start testing! 🚀

