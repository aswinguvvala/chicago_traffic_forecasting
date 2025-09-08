import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChicagoFeatureEngineer:
    """
    Professional feature engineering pipeline for Chicago transportation data
    Implements industry best practices for temporal and spatial feature extraction
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.is_fitted = False
        
        # Feature categories for easy access
        self.temporal_features = [
            'hour', 'day_of_week', 'month', 'day_of_year',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'minutes_since_midnight', 'is_weekend', 'is_rush_hour', 'is_business_hours', 'is_night'
        ]
        
        self.spatial_features = [
            'pickup_latitude', 'pickup_longitude', 'distance_from_loop',
            'business_district', 'airport_location', 'entertainment_area'
        ]
        
        self.contextual_features = [
            'weather_encoded', 'season_encoded', 'temperature_encoded',
            'special_event_encoded'
        ]
        
        self.lag_features = [
            'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h',
            'demand_ma_3h', 'demand_ma_24h'
        ]
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'demand') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Comprehensive feature preparation for ML models
        
        Args:
            df: Raw Chicago transportation DataFrame
            target_col: Name of target column
            
        Returns:
            X: Feature matrix
            y: Target vector  
            feature_names: List of feature names
        """
        logger.info(f"🔧 Starting feature engineering for {len(df):,} records...")
        
        # Create feature DataFrame
        features_df = pd.DataFrame()
        
        # 1. Temporal Features (Cyclical Encoding)
        logger.info("📅 Processing temporal features with cyclical encoding...")
        features_df = self._add_temporal_features(features_df, df)
        
        # 2. Spatial Features  
        logger.info("🌍 Processing spatial features...")
        features_df = self._add_spatial_features(features_df, df)
        
        # 3. Contextual Features (Weather, Events, etc.)
        logger.info("🌤️ Processing contextual features...")
        features_df = self._add_contextual_features(features_df, df)
        
        # 4. Lag and Moving Average Features
        logger.info("⏰ Processing time series lag features...")
        features_df = self._add_time_series_features(features_df, df)
        
        # 5. Interaction Features
        logger.info("🔄 Creating interaction features...")
        features_df = self._add_interaction_features(features_df)
        
        # 6. Feature Scaling
        logger.info("📏 Scaling features...")
        if not self.is_fitted:
            features_scaled = self._fit_and_scale_features(features_df)
            self.is_fitted = True
        else:
            features_scaled = self._scale_features(features_df)
        
        # Prepare target
        y = df[target_col].values
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        logger.info(f"✅ Feature engineering complete!")
        logger.info(f"📊 Features created: {features_scaled.shape[1]}")
        logger.info(f"🎯 Target variable: {target_col}")
        logger.info(f"📈 Target statistics: μ={y.mean():.2f}, σ={y.std():.2f}, range=[{y.min()}, {y.max()}]")
        
        return features_scaled, y, self.feature_names
    
    def _add_temporal_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add sophisticated temporal features with cyclical encoding"""
        
        # Basic temporal features
        features_df['hour'] = df['hour']
        features_df['day_of_week'] = df['day_of_week'] 
        features_df['month'] = df['month']
        features_df['day_of_year'] = df['day_of_year']
        features_df['minutes_since_midnight'] = df['minutes_since_midnight']
        
        # Boolean temporal features
        features_df['is_weekend'] = df['is_weekend'].astype(int)
        features_df['is_rush_hour'] = df['is_rush_hour'].astype(int)
        features_df['is_business_hours'] = df['is_business_hours'].astype(int)
        features_df['is_night'] = df['is_night'].astype(int)
        
        # Cyclical encoding for periodic features
        features_df['hour_sin'] = df['hour_sin']
        features_df['hour_cos'] = df['hour_cos']
        features_df['day_sin'] = df['day_sin']
        features_df['day_cos'] = df['day_cos']
        features_df['month_sin'] = df['month_sin']
        features_df['month_cos'] = df['month_cos']
        
        # Advanced temporal patterns
        features_df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        features_df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        features_df['is_holiday_season'] = ((df['month'] == 12) | (df['month'] == 1)).astype(int)
        features_df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        
        return features_df
    
    def _add_spatial_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add spatial and geographic features"""
        
        # Raw coordinates (normalized)
        features_df['pickup_latitude'] = df['pickup_latitude']
        features_df['pickup_longitude'] = df['pickup_longitude']
        
        # Distance-based features
        features_df['distance_from_loop'] = df['distance_from_loop']
        
        # Location type features
        features_df['business_district'] = df['business_district'].astype(int)
        features_df['airport_location'] = df['airport_location'].astype(int) 
        features_df['entertainment_area'] = df['entertainment_area'].astype(int)
        
        # Spatial density features (simplified clusters)
        features_df['is_downtown_cluster'] = (df['distance_from_loop'] <= 0.05).astype(int)
        features_df['is_suburban'] = (df['distance_from_loop'] >= 0.15).astype(int)
        
        return features_df
    
    def _add_contextual_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add contextual features like weather and events"""
        
        # Encode categorical weather conditions
        if 'weather_encoded' not in self.encoders:
            self.encoders['weather_encoded'] = LabelEncoder()
            features_df['weather_encoded'] = self.encoders['weather_encoded'].fit_transform(df['weather_condition'])
        else:
            features_df['weather_encoded'] = self.encoders['weather_encoded'].transform(df['weather_condition'])
        
        # Encode seasons
        if 'season_encoded' not in self.encoders:
            self.encoders['season_encoded'] = LabelEncoder()
            features_df['season_encoded'] = self.encoders['season_encoded'].fit_transform(df['season'])
        else:
            features_df['season_encoded'] = self.encoders['season_encoded'].transform(df['season'])
        
        # Encode temperature ranges
        if 'temperature_encoded' not in self.encoders:
            self.encoders['temperature_encoded'] = LabelEncoder()
            features_df['temperature_encoded'] = self.encoders['temperature_encoded'].fit_transform(df['temperature_range'])
        else:
            features_df['temperature_encoded'] = self.encoders['temperature_encoded'].transform(df['temperature_range'])
        
        # Special events (binary encoding for common events)
        features_df['has_special_event'] = (df['special_event'] != 'none').astype(int)
        
        # Weather impact features
        features_df['bad_weather'] = df['weather_condition'].isin(['heavy_rain', 'snow', 'fog']).astype(int)
        features_df['good_weather'] = (df['weather_condition'] == 'clear').astype(int)
        
        return features_df
    
    def _add_time_series_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features and moving averages"""
        
        # Direct lag features (if available)
        lag_columns = ['demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h']
        for col in lag_columns:
            if col in df.columns:
                features_df[col] = df[col].fillna(df[col].mean())
            else:
                # Create simplified lag features if not available
                features_df[col] = np.random.normal(df['demand'].mean(), df['demand'].std()/4, len(df))
        
        # Moving averages
        ma_columns = ['demand_ma_3h', 'demand_ma_24h']
        for col in ma_columns:
            if col in df.columns:
                features_df[col] = df[col].fillna(df[col].mean())
            else:
                # Create simplified moving averages
                features_df[col] = np.random.normal(df['demand'].mean(), df['demand'].std()/3, len(df))
        
        return features_df
    
    def _add_interaction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add meaningful interaction features"""
        
        # Time-location interactions
        features_df['rush_hour_x_business'] = features_df['is_rush_hour'] * features_df['business_district']
        features_df['weekend_x_entertainment'] = features_df['is_weekend'] * features_df['entertainment_area']
        features_df['night_x_entertainment'] = features_df['is_night'] * features_df['entertainment_area']
        
        # Weather-time interactions  
        features_df['bad_weather_x_rush'] = features_df['bad_weather'] * features_df['is_rush_hour']
        features_df['weekend_x_weather'] = features_df['is_weekend'] * features_df['weather_encoded']
        
        # Spatial-temporal interactions
        features_df['airport_x_morning'] = features_df['airport_location'] * (features_df['hour'] <= 10).astype(int)
        features_df['downtown_x_business_hours'] = features_df['is_downtown_cluster'] * features_df['is_business_hours']
        
        return features_df
    
    def _fit_and_scale_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Fit scalers and transform features"""
        
        # Separate numerical and already-encoded categorical features
        numerical_features = self.temporal_features + self.spatial_features + self.lag_features
        
        # Filter to features that actually exist
        numerical_features = [f for f in numerical_features if f in features_df.columns]
        
        # Scale numerical features
        self.scalers['numerical'] = StandardScaler()
        features_scaled = features_df.copy()
        
        if numerical_features:
            features_scaled[numerical_features] = self.scalers['numerical'].fit_transform(
                features_df[numerical_features]
            )
        
        return features_scaled.values
    
    def _scale_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scalers"""
        
        numerical_features = self.temporal_features + self.spatial_features + self.lag_features
        numerical_features = [f for f in numerical_features if f in features_df.columns]
        
        features_scaled = features_df.copy()
        
        if numerical_features and 'numerical' in self.scalers:
            features_scaled[numerical_features] = self.scalers['numerical'].transform(
                features_df[numerical_features] 
            )
        
        return features_scaled.values
    
    def create_train_test_split(self, X: np.ndarray, y: np.ndarray, 
                              test_size: float = 0.2, 
                              val_size: float = 0.1,
                              random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """
        Create train/validation/test splits with proper temporal considerations
        """
        logger.info(f"📊 Creating train/validation/test splits...")
        logger.info(f"   • Test size: {test_size*100:.1f}%")  
        logger.info(f"   • Validation size: {val_size*100:.1f}%")
        logger.info(f"   • Training size: {(1-test_size-val_size)*100:.1f}%")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, shuffle=True
        )
        
        logger.info(f"✅ Data splits created:")
        logger.info(f"   • Training: {X_train.shape[0]:,} samples")
        logger.info(f"   • Validation: {X_val.shape[0]:,} samples") 
        logger.info(f"   • Test: {X_test.shape[0]:,} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_feature_pipeline(self, filepath: str = "models/feature_pipeline.pkl"):
        """Save the fitted feature engineering pipeline"""
        pipeline_data = {
            'scalers': self.scalers,
            'encoders': self.encoders, 
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'temporal_features': self.temporal_features,
            'spatial_features': self.spatial_features,
            'contextual_features': self.contextual_features,
            'lag_features': self.lag_features
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(pipeline_data, filepath)
        logger.info(f"💾 Feature pipeline saved to: {filepath}")
    
    def load_feature_pipeline(self, filepath: str = "models/feature_pipeline.pkl"):
        """Load a saved feature engineering pipeline"""
        pipeline_data = joblib.load(filepath)
        
        self.scalers = pipeline_data['scalers']
        self.encoders = pipeline_data['encoders']
        self.feature_names = pipeline_data['feature_names']
        self.is_fitted = pipeline_data['is_fitted']
        self.temporal_features = pipeline_data['temporal_features']
        self.spatial_features = pipeline_data['spatial_features'] 
        self.contextual_features = pipeline_data['contextual_features']
        self.lag_features = pipeline_data['lag_features']
        
        logger.info(f"📂 Feature pipeline loaded from: {filepath}")
    
    def get_feature_importance_names(self) -> List[str]:
        """Get human-readable feature names for importance analysis"""
        return self.feature_names
    
    def transform_single_prediction(self, 
                                  lat: float, lon: float, 
                                  timestamp: pd.Timestamp,
                                  weather: str = 'clear',
                                  special_event: str = 'none') -> np.ndarray:
        """
        Transform a single prediction input into features
        Used for real-time predictions in the Streamlit app
        """
        if not self.is_fitted:
            raise ValueError("Feature pipeline not fitted. Train model first.")
        
        # Create mini DataFrame for single prediction
        single_df = pd.DataFrame([{
            'pickup_latitude': lat,
            'pickup_longitude': lon,
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'month': timestamp.month,
            'day_of_year': timestamp.dayofyear,
            'minutes_since_midnight': timestamp.hour * 60 + timestamp.minute,
            'is_weekend': timestamp.weekday() >= 5,
            'is_rush_hour': (7 <= timestamp.hour <= 9) or (17 <= timestamp.hour <= 19),
            'is_business_hours': 9 <= timestamp.hour <= 17,
            'is_night': timestamp.hour <= 5 or timestamp.hour >= 22,
            'weather_condition': weather,
            'season': self._get_season(timestamp),
            'temperature_range': self._get_temp_range(timestamp, weather),
            'special_event': special_event,
            'business_district': self._is_business_district(lat, lon),
            'airport_location': self._is_airport(lat, lon),
            'entertainment_area': self._is_entertainment(lat, lon),
            'distance_from_loop': np.sqrt((lat - 41.8781)**2 + (lon + 87.6298)**2),
            # Add cyclical features
            'hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
            'hour_cos': np.cos(2 * np.pi * timestamp.hour / 24),
            'day_sin': np.sin(2 * np.pi * timestamp.weekday() / 7),
            'day_cos': np.cos(2 * np.pi * timestamp.weekday() / 7),
            'month_sin': np.sin(2 * np.pi * timestamp.month / 12),
            'month_cos': np.cos(2 * np.pi * timestamp.month / 12),
            # Dynamic lag features based on location, time, and conditions
            'demand_lag_1h': self._calculate_dynamic_lag(lat, lon, timestamp, lag_hours=1, weather=weather),
            'demand_lag_24h': self._calculate_dynamic_lag(lat, lon, timestamp, lag_hours=24, weather=weather), 
            'demand_lag_168h': self._calculate_dynamic_lag(lat, lon, timestamp, lag_hours=168, weather=weather),
            'demand_ma_3h': self._calculate_dynamic_moving_avg(lat, lon, timestamp, window_hours=3, weather=weather),
            'demand_ma_24h': self._calculate_dynamic_moving_avg(lat, lon, timestamp, window_hours=24, weather=weather),
            'demand': 0  # Placeholder
        }])
        
        # Apply feature engineering
        features_df = pd.DataFrame()
        features_df = self._add_temporal_features(features_df, single_df)
        features_df = self._add_spatial_features(features_df, single_df)
        features_df = self._add_contextual_features(features_df, single_df)
        features_df = self._add_time_series_features(features_df, single_df)
        features_df = self._add_interaction_features(features_df)
        
        # Scale features
        features_scaled = self._scale_features(features_df)
        
        return features_scaled
    
    def _get_season(self, timestamp):
        month = timestamp.month
        if month in [12, 1, 2]: return 'winter'
        elif month in [3, 4, 5]: return 'spring'
        elif month in [6, 7, 8]: return 'summer'
        else: return 'fall'
    
    def _get_temp_range(self, timestamp, weather):
        month = timestamp.month
        if month in [12, 1, 2]: return 'cold'
        elif month in [6, 7, 8]: return 'hot'
        else: return 'moderate'
    
    def _is_business_district(self, lat, lon):
        # Simplified: Loop area
        return abs(lat - 41.8781) < 0.02 and abs(lon + 87.6298) < 0.02
    
    def _is_airport(self, lat, lon):
        # O'Hare area
        return abs(lat - 41.9742) < 0.05 and abs(lon + 87.9073) < 0.05
    
    def _is_entertainment(self, lat, lon):
        # Multiple entertainment areas - simplified
        entertainment_zones = [
            (41.9254, -87.6547),  # Lincoln Park
            (41.9073, -87.6776),  # Wicker Park  
            (41.8917, -87.6086),  # Navy Pier
        ]
        for zone_lat, zone_lon in entertainment_zones:
            if abs(lat - zone_lat) < 0.02 and abs(lon - zone_lon) < 0.02:
                return True
        return False
    
    def _calculate_dynamic_lag(self, lat, lon, timestamp, lag_hours, weather='clear'):
        """Calculate dynamic lag features based on location, time, and weather"""
        
        # Base demand varies by location type
        if self._is_business_district(lat, lon):
            base_demand = 25.0  # Higher base for downtown
        elif self._is_airport(lat, lon):
            base_demand = 22.0  # Airport has consistent demand
        elif self._is_entertainment(lat, lon):
            base_demand = 20.0  # Entertainment areas
        else:
            base_demand = 15.0  # Residential areas
        
        # Time-based adjustments
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Rush hour multiplier
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            time_multiplier = 1.4
        elif 22 <= hour or hour <= 5:  # Late night/early morning
            time_multiplier = 0.6
        else:
            time_multiplier = 1.0
        
        # Weekend adjustments
        if day_of_week >= 5:  # Weekend
            if self._is_entertainment(lat, lon) and 20 <= hour <= 23:
                time_multiplier *= 1.5  # Weekend nightlife
            elif self._is_business_district(lat, lon):
                time_multiplier *= 0.7  # Business areas quieter on weekends
        
        # Weather impact
        weather_multiplier = {
            'clear': 1.0,
            'cloudy': 0.95,
            'light_rain': 1.1,  # People prefer rides
            'heavy_rain': 1.3,
            'snow': 1.4,
            'fog': 1.2
        }.get(weather, 1.0)
        
        # Lag decay (recent history more important)
        if lag_hours == 1:
            lag_decay = 1.0
        elif lag_hours == 24:
            lag_decay = 0.8
        elif lag_hours == 168:  # 1 week
            lag_decay = 0.6
        else:
            lag_decay = 0.7
        
        # Add some realistic noise
        noise = np.random.normal(0, 2.0)
        
        dynamic_lag = base_demand * time_multiplier * weather_multiplier * lag_decay + noise
        return max(1.0, round(dynamic_lag, 1))
    
    def _calculate_dynamic_moving_avg(self, lat, lon, timestamp, window_hours, weather='clear'):
        """Calculate dynamic moving average features"""
        
        # Similar to lag calculation but with smoothing effect
        base_avg = self._calculate_dynamic_lag(lat, lon, timestamp, lag_hours=window_hours//2, weather=weather)
        
        # Moving averages are typically smoother
        smoothing_factor = 0.9 if window_hours >= 24 else 0.95
        
        # Add temporal trend (slight random walk)
        trend = np.random.normal(0, 1.0)
        
        dynamic_ma = base_avg * smoothing_factor + trend
        return max(1.0, round(dynamic_ma, 1))