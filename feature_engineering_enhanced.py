"""
Enhanced Feature Engineering for Land Property Value Prediction
Adds comprehensive features for better model accuracy
"""

import pandas as pd
import numpy as np
from datetime import datetime
import math

class FeatureEngineering:
    """
    Feature engineering class for land property value prediction
    """
    
    def __init__(self):
        # Reference points for distance calculations (can be customized)
        # These should be set to your city center or important landmarks
        self.city_center_lat = 14.5995  # Example: Manila coordinates
        self.city_center_lon = 120.9842
        
        # Amenity locations (can be loaded from database or config)
        self.amenity_locations = {
            'schools': [],
            'hospitals': [],
            'shopping_malls': [],
            'parks': []
        }
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between two points using Haversine formula
        Returns distance in kilometers
        """
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            return None
        
        try:
            lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
            
            # Haversine formula
            R = 6371  # Earth radius in kilometers
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (math.sin(dlat/2) * math.sin(dlat/2) +
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                 math.sin(dlon/2) * math.sin(dlon/2))
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = R * c
            return distance
        except:
            return None
    
    def engineer_features(self, df):
        """
        Main feature engineering function
        Adds all engineered features to the dataframe
        """
        if df is None or len(df) == 0:
            return df
        
        df = df.copy()
        
        # 1. SIZE FEATURES
        df = self._add_size_features(df)
        
        # 2. LOCATION FEATURES
        df = self._add_location_features(df)
        
        # 3. TEMPORAL FEATURES
        df = self._add_temporal_features(df)
        
        # 4. ZONING/TYPE FEATURES
        df = self._add_zoning_features(df)
        
        # 5. INTERACTION FEATURES
        df = self._add_interaction_features(df)
        
        # 6. MARKET FEATURES (if data available)
        df = self._add_market_features(df)
        
        return df
    
    def _add_size_features(self, df):
        """Add size-related features"""
        # Area ratio (building density)
        df['area_ratio'] = df['project_area'] / (df['lot_area'] + 1e-6)
        df['area_ratio'] = df['area_ratio'].clip(0, 1)  # Cap at 1
        
        # Lot size categories
        df['lot_size_category'] = pd.cut(
            df['lot_area'],
            bins=[0, 100, 300, 500, 1000, float('inf')],
            labels=['very_small', 'small', 'medium', 'large', 'very_large']
        )
        
        # Project area categories
        df['project_size_category'] = pd.cut(
            df['project_area'],
            bins=[0, 50, 150, 300, 500, float('inf')],
            labels=['very_small', 'small', 'medium', 'large', 'very_large']
        )
        
        # Size difference
        df['size_difference'] = df['lot_area'] - df['project_area']
        
        # Efficiency ratio
        df['efficiency_ratio'] = df['project_area'] / (df['lot_area'] + 1e-6)
        
        return df
    
    def _add_location_features(self, df):
        """Add location-based features"""
        # Distance to city center (if lat/lon available)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['distance_to_center'] = df.apply(
                lambda row: self.calculate_distance(
                    row['latitude'], row['longitude'],
                    self.city_center_lat, self.city_center_lon
                ) if pd.notna(row['latitude']) and pd.notna(row['longitude']) else None,
                axis=1
            )
            
            # Distance categories
            df['distance_category'] = pd.cut(
                df['distance_to_center'],
                bins=[0, 5, 10, 20, 50, float('inf')],
                labels=['city_center', 'urban', 'suburban', 'rural', 'remote'],
                include_lowest=True
            )
        
        # Location type encoding (if available)
        if 'location_type' in df.columns:
            # Create location type categories
            location_mapping = {
                'Downtown': 'premium',
                'Urban Core': 'premium',
                'Commercial District': 'premium',
                'Suburban': 'standard',
                'Residential Area': 'standard',
                'Mixed Zone': 'standard',
                'Industrial Zone': 'economy',
                'Rural': 'economy',
                'Agricultural': 'economy'
            }
            df['location_category'] = df['project_location'].map(location_mapping).fillna('standard')
        
        # Location frequency (how common is this location)
        if 'project_location' in df.columns:
            location_counts = df['project_location'].value_counts()
            df['location_frequency'] = df['project_location'].map(location_counts)
        
        return df
    
    def _add_temporal_features(self, df):
        """Add time-based features"""
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Year and month (already have, but ensure they exist)
            df['year'] = df['created_at'].dt.year
            df['month'] = df['created_at'].dt.month
            
            # Day of week
            df['day_of_week'] = df['created_at'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Season
            df['season'] = df['month'].apply(self._get_season)
            
            # Quarter
            df['quarter'] = df['created_at'].dt.quarter
            
            # Days since reference date (for trend analysis)
            reference_date = df['created_at'].min()
            df['days_since_reference'] = (df['created_at'] - reference_date).dt.days
            
            # Year-month combination
            df['year_month'] = df['created_at'].dt.to_period('M')
            
            # Cyclical encoding for month (captures seasonality)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Cyclical encoding for day of week
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _get_season(self, month):
        """Get season from month"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _add_zoning_features(self, df):
        """Add zoning and type-related features"""
        # Project type combinations
        if 'project_type' in df.columns and 'project_nature' in df.columns:
            df['type_nature_combo'] = df['project_type'].astype(str) + '_' + df['project_nature'].astype(str)
        
        # Zoning type (if available)
        if 'site_zoning' in df.columns:
            # Create zoning categories
            zoning_premium = ['Commercial', 'Mixed-Use', 'Residential-High']
            zoning_standard = ['Residential', 'Institutional']
            zoning_economy = ['Agricultural', 'Industrial', 'Rural']
            
            df['zoning_category'] = df['site_zoning'].apply(
                lambda x: 'premium' if x in zoning_premium
                else 'standard' if x in zoning_standard
                else 'economy' if x in zoning_economy
                else 'unknown'
            )
        
        # Land use diversity (if multiple land uses)
        if 'land_uses' in df.columns:
            df['land_use_count'] = df['land_uses'].apply(
                lambda x: len(str(x).split(',')) if pd.notna(x) else 0
            )
        
        return df
    
    def _add_interaction_features(self, df):
        """Add interaction features between variables"""
        # Size × Location interaction
        if 'lot_area' in df.columns and 'location_category' in df.columns:
            df['size_location_interaction'] = df['lot_area'] * df['location_category'].map({
                'premium': 1.5,
                'standard': 1.0,
                'economy': 0.7
            }).fillna(1.0)
        
        # Type × Size interaction
        if 'project_type' in df.columns and 'lot_area' in df.columns:
            # Commercial projects on large lots are more valuable
            df['type_size_interaction'] = df.apply(
                lambda row: row['lot_area'] * 1.3 if row['project_type'] == 'commercial'
                else row['lot_area'] * 1.1 if row['project_type'] == 'mixed-use'
                else row['lot_area'],
                axis=1
            )
        
        # Year × Location interaction (captures location appreciation over time)
        if 'year' in df.columns and 'location_category' in df.columns:
            base_year = df['year'].min()
            df['year_location_interaction'] = (df['year'] - base_year) * df['location_category'].map({
                'premium': 1.2,
                'standard': 1.0,
                'economy': 0.8
            }).fillna(1.0)
        
        return df
    
    def _add_market_features(self, df):
        """Add market condition features (if data available)"""
        # These would require external data sources or database tables
        
        # Example: Inflation adjustment (if you have inflation data)
        # df['inflation_adjusted_year'] = ...
        
        # Example: Market trend (if you have historical price data)
        # df['market_trend'] = ...
        
        # For now, we'll add placeholder features that can be enhanced later
        if 'year' in df.columns:
            # Simple time trend
            base_year = df['year'].min()
            df['years_since_base'] = df['year'] - base_year
            
            # Assume 3% annual inflation (can be replaced with actual data)
            df['inflation_factor'] = 1.03 ** (df['years_since_base'])
        
        return df
    
    def get_feature_list(self):
        """Get list of all engineered features"""
        return [
            # Size features
            'area_ratio', 'lot_size_category', 'project_size_category',
            'size_difference', 'efficiency_ratio',
            
            # Location features
            'distance_to_center', 'distance_category', 'location_category',
            'location_frequency',
            
            # Temporal features
            'day_of_week', 'is_weekend', 'season', 'quarter',
            'days_since_reference', 'month_sin', 'month_cos',
            'day_sin', 'day_cos',
            
            # Zoning features
            'type_nature_combo', 'zoning_category', 'land_use_count',
            
            # Interaction features
            'size_location_interaction', 'type_size_interaction',
            'year_location_interaction',
            
            # Market features
            'years_since_base', 'inflation_factor'
        ]


def calculate_proximity_features(lat, lon, amenity_locations):
    """
    Calculate proximity to various amenities
    Returns dict with distances to each amenity type
    """
    if pd.isna(lat) or pd.isna(lon):
        return {}
    
    features = {}
    
    for amenity_type, locations in amenity_locations.items():
        if len(locations) > 0:
            distances = []
            for amenity_lat, amenity_lon in locations:
                dist = calculate_distance(lat, lon, amenity_lat, amenity_lon)
                if dist is not None:
                    distances.append(dist)
            
            if distances:
                features[f'distance_to_{amenity_type}'] = min(distances)
                features[f'nearest_{amenity_type}_distance'] = min(distances)
                features[f'{amenity_type}_count_within_5km'] = sum(1 for d in distances if d <= 5)
                features[f'{amenity_type}_count_within_10km'] = sum(1 for d in distances if d <= 10)
    
    return features


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points (Haversine formula)"""
    try:
        R = 6371  # Earth radius in km
        dlat = math.radians(float(lat2) - float(lat1))
        dlon = math.radians(float(lon2) - float(lon1))
        a = (math.sin(dlat/2) * math.sin(dlat/2) +
             math.cos(math.radians(float(lat1))) * math.cos(math.radians(float(lat2))) *
             math.sin(dlon/2) * math.sin(dlon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    except:
        return None

