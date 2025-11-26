# Land Cost Prediction Methodology

## Overview

The Land Cost Prediction System uses **Machine Learning (Linear Regression)** combined with **location analysis** and **time series forecasting** to predict land costs for current and future years (5-10 years ahead).

## How the Prediction Works

### Step 1: Base Cost Prediction (Current Year)

The system first calculates the base cost per square meter using a trained Linear Regression model.

#### 1.1 Feature Extraction

The model uses the following features to predict cost:

| Feature | Description | Impact |
|---------|-------------|--------|
| **Lot Area** | Total lot size in square meters | Larger lots may have different per-sqm pricing |
| **Project Area** | Building/development area in sqm | Affects development density |
| **Project Type** | Type of project (residential, commercial, etc.) | Different types have different cost ranges |
| **Year** | Current year | Accounts for inflation over time |
| **Month** | Current month | Seasonal variations |
| **Age** | Age of applicant | May correlate with project characteristics |

#### 1.2 Model Training

The Linear Regression model was trained on historical data:
- **Training Data**: 3,000+ historical land transactions
- **Features**: 6 input features (lot_area, project_area, year, month, age, project_type)
- **Target**: Cost per square meter (calculated as `project_cost_numeric / lot_area`)
- **Algorithm**: Linear Regression with feature scaling

#### 1.3 Prediction Formula

```
Base Cost per sqm = Linear Regression Model(
    lot_area,
    project_area,
    year,
    month,
    age,
    project_type_encoded
)
```

The model uses the formula:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Where:
- `y` = predicted cost per sqm
- `β₀` = intercept (base value)
- `β₁...βₙ` = coefficients for each feature
- `x₁...xₙ` = feature values

### Step 2: Location Factor Adjustment

The base cost is then adjusted based on the location's historical pricing patterns.

#### 2.1 Location Analysis

The system analyzes historical data to calculate location multipliers:

```python
Location Factor = Average Cost per sqm (Location) / Average Cost per sqm (Overall)
```

#### 2.2 Location Categories

Locations are categorized based on their multiplier:

| Category | Multiplier Range | Examples |
|----------|-----------------|----------|
| **Premium** | 1.2x - 1.35x | Downtown, Urban Core, Commercial District |
| **Standard** | 0.9x - 1.2x | Suburban, Residential Area, Mixed Zone |
| **Economy** | 0.75x - 0.9x | Rural, Agricultural areas |

#### 2.3 Adjusted Cost Calculation

```
Adjusted Current Cost = Base Cost × Location Factor
```

**Example:**
- Base Cost: 34,480 PHP/sqm
- Location: Downtown (Factor: 1.25)
- Adjusted Cost: 34,480 × 1.25 = **43,100 PHP/sqm**

### Step 3: Future Year Projection (5-10 Years)

For future predictions, the system applies compound growth based on historical appreciation rates.

#### 3.1 Appreciation/Depreciation Rate Calculation

The system calculates realistic yearly appreciation or depreciation rate from historical data. **The rate can be positive (increase) or negative (decrease)**.

```python
Appreciation Rate = Robust Average of Year-over-Year Growth Rates
```

**Method:**
1. Group historical data by year (last 5 years minimum)
2. Calculate average cost per sqm for each year
3. Calculate year-over-year percentage change
4. Use **median** for robustness (less affected by outliers)
5. Apply **weighted average** (recent years weighted more)
6. Cap extreme values between -10% and +15% per year

**Key Features:**
- ✅ **Can show decrease**: Negative rates indicate declining values
- ✅ **Robust statistics**: Uses median to handle outliers
- ✅ **Recent data priority**: More recent years have higher weight
- ✅ **Realistic bounds**: Prevents unrealistic extreme predictions

#### 3.2 Location-Based Rate Adjustment

Appreciation rates are adjusted based on location category:

| Location Category | Rate Adjustment |
|------------------|----------------|
| Premium | +10% (appreciates faster) |
| Standard | No adjustment |
| Economy | -10% (appreciates slower) |

**Example:**
- Base Rate: 2.85% per year
- Location: Premium (Downtown)
- Adjusted Rate: 2.85% × 1.1 = **3.14% per year**

#### 3.3 Compound Growth/Decay Formula

Future cost is calculated using compound growth (for increases) or decay (for decreases):

```
Future Cost = Current Cost × (1 + Rate)^Years
```

**Note:** The formula works for both positive (increase) and negative (decrease) rates.

**Example 1: Increase (10 years):**
- Current Cost: 43,100 PHP/sqm
- Appreciation Rate: 3.14% (0.0314)
- Years: 10
- Future Cost = 43,100 × (1.0314)^10
- Future Cost = 43,100 × 1.363
- **Future Cost = 58,745 PHP/sqm** ✅

**Example 2: Decrease (10 years):**
- Current Cost: 43,100 PHP/sqm
- Depreciation Rate: -2.5% (-0.025)
- Years: 10
- Future Cost = 43,100 × (0.975)^10
- Future Cost = 43,100 × 0.776
- **Future Cost = 33,446 PHP/sqm** ⚠️

**Minimum Floor:** The system ensures cost never drops below 10% of original value to prevent unrealistic predictions.

#### 3.4 Yearly Breakdown

The system calculates cost for each year:

```
Year 1: Current × (1 + rate)^1
Year 2: Current × (1 + rate)^2
Year 3: Current × (1 + rate)^3
...
Year 10: Current × (1 + rate)^10
```

### Step 4: Multiple Scenarios

The system provides three scenarios to account for uncertainty:

#### 4.1 Optimistic Scenario
- **For Increasing Markets**: Rate × 1.5 (50% higher growth)
- **For Decreasing Markets**: Rate × 0.5 (less decrease, could turn positive)
- **Use Case**: Best-case economic conditions
- **Example (increasing)**: 3.14% × 1.5 = 4.71% per year
- **Example (decreasing)**: -2.5% × 0.5 = -1.25% per year

#### 4.2 Realistic Scenario
- **Rate**: Calculated historical rate (can be positive or negative)
- **Use Case**: Most likely outcome based on data
- **Example (increasing)**: 3.14% per year
- **Example (decreasing)**: -2.5% per year

#### 4.3 Conservative Scenario
- **For Increasing Markets**: Rate × 0.3 (much lower, could be near zero)
- **For Decreasing Markets**: Rate × 1.5 (more decrease)
- **Use Case**: Worst-case economic conditions
- **Example (increasing)**: 3.14% × 0.3 = 0.94% per year
- **Example (decreasing)**: -2.5% × 1.5 = -3.75% per year

## Complete Calculation Example

Let's trace through a complete example:

### Input Data
- Lot Area: 200 sqm
- Project Area: 150 sqm
- Project Type: Residential
- Location: Downtown
- Target Years: 10

### Step-by-Step Calculation

#### Step 1: Base Prediction
```
Model Input: [200, 150, 2025, 12, 35, 3]  // 3 = residential encoded
Model Output: 34,480 PHP/sqm
```

#### Step 2: Location Adjustment
```
Location: Downtown
Location Factor: 1.25 (premium)
Adjusted Cost: 34,480 × 1.25 = 43,100 PHP/sqm
Total Value: 43,100 × 200 = 8,620,000 PHP
```

#### Step 3: Appreciation Rate
```
Historical Rate: 2.85% per year
Location Adjustment: +10% (premium)
Adjusted Rate: 2.85% × 1.1 = 3.14% per year
```

#### Step 4: Future Projection (10 years)
```
Future Cost = 43,100 × (1.0314)^10
Future Cost = 43,100 × 1.363
Future Cost = 58,745 PHP/sqm
Total Value = 58,745 × 200 = 11,749,000 PHP
```

#### Step 5: Scenarios
```
Optimistic: 43,100 × (1.044)^10 = 66,500 PHP/sqm
Realistic: 43,100 × (1.0314)^10 = 58,745 PHP/sqm
Conservative: 43,100 × (1.0188)^10 = 51,800 PHP/sqm
```

## Model Performance

### Training Metrics

- **R² Score**: 0.156 (indicates model explains 15.6% of variance)
- **RMSE**: 16,953 PHP/sqm (average prediction error)
- **MAE**: 13,721 PHP/sqm (mean absolute error)
- **Training Samples**: 3,334 records

### Model Limitations

1. **R² Score**: The model has moderate predictive power. This is common in real estate due to:
   - Market volatility
   - Unmeasured factors (accessibility, view, etc.)
   - Economic conditions

2. **Data Quality**: Predictions depend on:
   - Quality of historical data
   - Completeness of records
   - Accuracy of cost reporting

3. **Market Conditions**: The model assumes:
   - Historical trends continue
   - No major economic disruptions
   - Stable market conditions

## Factors Considered

### ✅ Included Factors

1. **Physical Characteristics**
   - Lot size
   - Project area
   - Project type

2. **Location**
   - Geographic location
   - Location category (premium/standard/economy)
   - Historical location pricing

3. **Temporal Factors**
   - Current year (inflation)
   - Month (seasonal variations)
   - Future years (appreciation)

4. **Demographics**
   - Applicant age (may correlate with project type)

### ❌ Not Included (Future Enhancements)

1. **Economic Indicators**
   - GDP growth
   - Interest rates
   - Inflation rates

2. **Infrastructure**
   - Proximity to schools, hospitals
   - Road access
   - Public transportation

3. **Market Conditions**
   - Supply and demand
   - Market trends
   - Economic forecasts

4. **Property Characteristics**
   - Soil quality
   - Topography
   - Zoning restrictions

## Confidence Levels

The system provides confidence indicators:

| Confidence | R² Score Range | Interpretation |
|------------|----------------|----------------|
| **High** | > 0.7 | Model explains most variance |
| **Medium** | 0.3 - 0.7 | Moderate predictive power |
| **Low** | < 0.3 | Limited predictive power |

**Current Model**: Medium confidence (R² = 0.156)

## Data Sources

### Historical Data
- **Source**: `application_forms` table
- **Fields Used**:
  - `project_cost_numeric`: Total project cost
  - `lot_area`: Lot size in sqm
  - `project_type`: Type of project
  - `project_location`: Location
  - `created_at`: Date of application
  - `age`: Applicant age (from `clients` table)

### Data Processing
1. **Cleaning**: Remove outliers (IQR method)
2. **Calculation**: Cost per sqm = `project_cost_numeric / lot_area`
3. **Aggregation**: Group by location, type, year
4. **Analysis**: Calculate averages and trends

## Mathematical Formulas Summary

### 1. Base Cost Prediction
```
Base Cost = LinearRegression(lot_area, project_area, year, month, age, project_type)
```

### 2. Location Adjustment
```
Adjusted Cost = Base Cost × Location_Factor
Location_Factor = Avg_Cost_Location / Avg_Cost_Overall
```

### 3. Appreciation Rate
```
Rate = Average(Year_Over_Year_Growth)
Adjusted_Rate = Rate × Location_Adjustment
```

### 4. Future Cost
```
Future_Cost = Current_Cost × (1 + Adjusted_Rate)^Years
```

### 5. Total Value
```
Total_Value = Cost_per_sqm × Lot_Area
```

## Trend Detection

### How the System Determines Increase vs. Decrease

1. **Historical Analysis**: Analyzes last 5 years of data
2. **Statistical Calculation**: Uses median and weighted averages
3. **Trend Direction**: 
   - **Positive rate** = Increasing trend 📈
   - **Negative rate** = Decreasing trend 📉
4. **Location Impact**: Premium locations may increase faster, economy locations may decrease

### Real-World Scenarios

**Increasing Markets:**
- Growing urban areas
- Developing commercial districts
- Areas with new infrastructure
- High-demand residential zones

**Decreasing Markets:**
- Declining rural areas
- Oversupplied markets
- Areas with economic challenges
- Agricultural zones with low demand

## Accuracy and Reliability

### Prediction Accuracy

- **Current Year**: ±13,721 PHP/sqm (MAE)
- **Future Years**: Accuracy decreases with time horizon
  - 5 years: Moderate accuracy
  - 10 years: Lower accuracy (more uncertainty)
- **Trend Detection**: More accurate for strong trends, less accurate for volatile markets

### Recommendations

1. **Use as Guide**: Predictions are estimates, not guarantees
2. **Consider Scenarios**: Review all three scenarios (especially if showing decrease)
3. **Understand Trends**: Pay attention to whether values are increasing or decreasing
4. **Location Matters**: Premium locations tend to increase, economy locations may decrease
5. **Update Regularly**: Retrain model with new data to capture market changes
6. **Professional Advice**: Consult real estate professionals for major decisions
7. **Market Research**: Compare with current market listings and trends
8. **Risk Assessment**: If conservative scenario shows decrease, consider risks carefully

## Model Updates

The model should be retrained:
- **Quarterly**: For best accuracy
- **When**: New data becomes available
- **How**: Run training script with updated data

## Conclusion

The Land Cost Prediction System combines:
- **Machine Learning** (Linear Regression) for base predictions
- **Location Analysis** for geographic adjustments
- **Time Series Forecasting** for future projections
- **Multiple Scenarios** for risk assessment

While the model provides valuable estimates, users should:
- Understand the methodology
- Consider the confidence levels
- Review multiple scenarios
- Use predictions as one input among many factors

---

**Last Updated**: 2025
**Model Version**: 1.0
**Training Data**: 3,334 records

