import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChicagoTransportationDataGenerator:
    """
    Generate realistic Chicago transportation data with proper patterns
    Based on actual Chicago transportation dynamics and geographic data
    """
    
    def __init__(self):
        # Chicago neighborhoods with real coordinates and demand weights
        self.neighborhoods = {
            'Loop': {
                'coords': (41.8781, -87.6298),
                'demand_weight': 0.35,
                'business_district': True,
                'airport': False,
                'entertainment': False
            },
            'Lincoln Park': {
                'coords': (41.9254, -87.6547),
                'demand_weight': 0.25,
                'business_district': False,
                'airport': False,
                'entertainment': True
            },
            'O\'Hare Airport': {
                'coords': (41.9742, -87.9073),
                'demand_weight': 0.40,
                'business_district': False,
                'airport': True,
                'entertainment': False
            },
            'Wicker Park': {
                'coords': (41.9073, -87.6776),
                'demand_weight': 0.22,
                'business_district': False,
                'airport': False,
                'entertainment': True
            },
            'Hyde Park': {
                'coords': (41.7943, -87.5907),
                'demand_weight': 0.18,
                'business_district': False,
                'airport': False,
                'entertainment': False
            },
            'Navy Pier': {
                'coords': (41.8917, -87.6086),
                'demand_weight': 0.28,
                'business_district': False,
                'airport': False,
                'entertainment': True
            },
            'Magnificent Mile': {
                'coords': (41.8955, -87.6244),
                'demand_weight': 0.32,
                'business_district': True,
                'airport': False,
                'entertainment': True
            },
            'Lakeview': {
                'coords': (41.9403, -87.6438),
                'demand_weight': 0.20,
                'business_district': False,
                'airport': False,
                'entertainment': True
            },
            'Logan Square': {
                'coords': (41.9294, -87.7073),
                'demand_weight': 0.19,
                'business_district': False,
                'airport': False,
                'entertainment': True
            },
            'Chinatown': {
                'coords': (41.8508, -87.6320),
                'demand_weight': 0.15,
                'business_district': False,
                'airport': False,
                'entertainment': True
            }
        }
        
        # Weather conditions with impact on demand
        self.weather_conditions = {
            'clear': {'probability': 0.35, 'demand_multiplier': 1.0},
            'cloudy': {'probability': 0.25, 'demand_multiplier': 1.1},
            'light_rain': {'probability': 0.15, 'demand_multiplier': 1.4},
            'heavy_rain': {'probability': 0.10, 'demand_multiplier': 1.8},
            'snow': {'probability': 0.10, 'demand_multiplier': 2.2},
            'fog': {'probability': 0.05, 'demand_multiplier': 1.3}
        }
        
        # Special events that impact demand
        self.special_events = {
            'Cubs Game': {'locations': ['Lakeview'], 'multiplier': 2.5, 'duration_hours': 4},
            'Bulls Game': {'locations': ['Loop'], 'multiplier': 2.2, 'duration_hours': 3},
            'Concert': {'locations': ['Loop', 'Navy Pier'], 'multiplier': 1.8, 'duration_hours': 3},
            'Festival': {'locations': ['Lincoln Park', 'Wicker Park'], 'multiplier': 1.6, 'duration_hours': 8},
            'Conference': {'locations': ['Loop', 'Magnificent Mile'], 'multiplier': 1.4, 'duration_hours': 10}
        }
    
    def generate_realistic_dataset(self, 
                                 num_records: int = 100000,
                                 start_date: datetime = None,
                                 end_date: datetime = None) -> pd.DataFrame:
        """
        Generate comprehensive realistic Chicago transportation dataset
        """
        logger.info(f"🚀 Generating {num_records:,} realistic Chicago transportation records...")
        
        if start_date is None:
            start_date = datetime(2024, 1, 1)
        if end_date is None:
            end_date = datetime(2024, 12, 31)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        data = []
        
        for i in range(num_records):
            if i % 10000 == 0:
                logger.info(f"Generated {i:,} records...")
            
            # Generate timestamp with realistic distribution
            timestamp = self._generate_timestamp(start_date, end_date)
            
            # Select location based on demand patterns
            location_name = self._select_location(timestamp)
            location_data = self.neighborhoods[location_name]
            
            # Generate precise coordinates with realistic noise
            lat, lon = self._generate_coordinates(location_data['coords'])
            
            # Calculate base demand for this time and location
            base_demand = self._calculate_base_demand(timestamp, location_data)
            
            # Apply weather effects
            weather = self._select_weather(timestamp)
            weather_multiplier = self.weather_conditions[weather]['demand_multiplier']
            
            # Apply special event effects
            event, event_multiplier = self._check_special_events(timestamp, location_name)
            
            # Calculate final demand with noise
            final_demand = max(1, int(base_demand * weather_multiplier * event_multiplier + 
                                    np.random.normal(0, 3)))
            
            # Calculate business metrics
            trip_distance = self._generate_trip_distance(location_data)
            fare_amount = self._calculate_fare(trip_distance, final_demand)
            
            # Extract temporal features
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            month = timestamp.month
            is_weekend = day_of_week >= 5
            is_rush_hour = (7 <= hour <= 9) or (17 <= hour <= 19)
            is_business_hours = 9 <= hour <= 17
            is_night = hour <= 5 or hour >= 22
            
            # Create record
            record = {
                'trip_id': f"CHI_{i+1:06d}",
                'timestamp': timestamp,
                'pickup_latitude': lat,
                'pickup_longitude': lon,
                'location_name': location_name,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'year': timestamp.year,
                'is_weekend': is_weekend,
                'is_rush_hour': is_rush_hour,
                'is_business_hours': is_business_hours,
                'is_night': is_night,
                'weather_condition': weather,
                'special_event': event if event else 'none',
                'demand': final_demand,
                'trip_distance_miles': trip_distance,
                'fare_amount': fare_amount,
                'business_district': location_data['business_district'],
                'airport_location': location_data['airport'],
                'entertainment_area': location_data['entertainment'],
                'season': self._get_season(timestamp),
                'distance_from_loop': self._calculate_distance_from_loop(lat, lon),
                'temperature_range': self._get_temperature_range(timestamp, weather)
            }
            
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        logger.info(f"✅ Generated {len(df):,} records successfully!")
        logger.info(f"📊 Dataset Statistics:")
        logger.info(f"   • Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"   • Unique Locations: {df['location_name'].nunique()}")
        logger.info(f"   • Average Demand: {df['demand'].mean():.2f} rides/hour")
        logger.info(f"   • Weather Conditions: {df['weather_condition'].nunique()}")
        logger.info(f"   • Special Events: {len(df[df['special_event'] != 'none'])} records")
        
        return df
    
    def _generate_timestamp(self, start_date: datetime, end_date: datetime) -> datetime:
        """Generate timestamp with realistic hourly distribution"""
        # More rides during peak hours
        hour_weights = np.array([
            0.3, 0.2, 0.1, 0.1, 0.2, 0.4,  # 0-5 AM (low)
            0.8, 1.8, 2.0, 1.5, 1.2, 1.4,  # 6-11 AM (morning peak)
            1.6, 1.3, 1.1, 1.0, 1.2, 1.8,  # 12-5 PM (business hours)
            2.2, 1.9, 1.4, 1.0, 0.8, 0.5   # 6-11 PM (evening peak + night)
        ])
        hour_weights = hour_weights / hour_weights.sum()
        
        # Select random date in range
        days_range = (end_date - start_date).days
        random_days = np.random.randint(0, days_range)
        base_date = start_date + timedelta(days=random_days)
        
        # Select hour based on distribution
        hour = np.random.choice(24, p=hour_weights)
        minute = np.random.randint(0, 60)
        
        return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    def _select_location(self, timestamp: datetime) -> str:
        """Select location based on time-dependent demand patterns"""
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        
        # Adjust location probabilities based on time
        location_probs = []
        location_names = list(self.neighborhoods.keys())
        
        for location_name in location_names:
            location_data = self.neighborhoods[location_name]
            base_weight = location_data['demand_weight']
            
            # Business district boost during business hours on weekdays
            if location_data['business_district'] and 9 <= hour <= 17 and not is_weekend:
                base_weight *= 1.8
            
            # Entertainment area boost during evenings and weekends
            if location_data['entertainment'] and (hour >= 18 or is_weekend):
                base_weight *= 1.5
            
            # Airport consistent demand with flight-time patterns
            if location_data['airport']:
                if 6 <= hour <= 10 or 16 <= hour <= 20:  # Flight departure times
                    base_weight *= 1.3
            
            location_probs.append(base_weight)
        
        # Normalize probabilities
        location_probs = np.array(location_probs)
        location_probs = location_probs / location_probs.sum()
        
        return np.random.choice(location_names, p=location_probs)
    
    def _generate_coordinates(self, center_coords: Tuple[float, float]) -> Tuple[float, float]:
        """Generate realistic coordinates with noise around center point"""
        lat, lon = center_coords
        
        # Add realistic noise (roughly 1km radius)
        lat_noise = np.random.normal(0, 0.008)  # ~0.008 degrees ≈ 1km
        lon_noise = np.random.normal(0, 0.010)  # Adjust for Chicago's longitude
        
        return round(lat + lat_noise, 6), round(lon + lon_noise, 6)
    
    def _calculate_base_demand(self, timestamp: datetime, location_data: Dict) -> float:
        """Calculate base demand based on time and location characteristics"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = day_of_week >= 5
        
        # Base demand patterns
        if 7 <= hour <= 9:  # Morning rush
            base_demand = 45
        elif 17 <= hour <= 19:  # Evening rush
            base_demand = 50
        elif 10 <= hour <= 16:  # Business hours
            base_demand = 25
        elif 20 <= hour <= 23:  # Evening entertainment
            base_demand = 30
        elif 0 <= hour <= 2 and is_weekend:  # Weekend late night
            base_demand = 35
        else:  # Night/early morning
            base_demand = 12
        
        # Weekend adjustments
        if is_weekend:
            if 10 <= hour <= 14:  # Weekend afternoon
                base_demand *= 1.4
            elif hour <= 8:  # Weekend morning
                base_demand *= 0.6
        
        # Location-specific adjustments
        base_demand *= location_data['demand_weight']
        
        return base_demand
    
    def _select_weather(self, timestamp: datetime) -> str:
        """Select weather condition with seasonal patterns"""
        month = timestamp.month
        
        # Seasonal weather adjustments
        if month in [12, 1, 2]:  # Winter
            weather_probs = [0.2, 0.3, 0.15, 0.1, 0.2, 0.05]  # More snow
        elif month in [6, 7, 8]:  # Summer
            weather_probs = [0.5, 0.3, 0.1, 0.05, 0.02, 0.03]  # More clear weather
        else:  # Spring/Fall
            weather_probs = [0.35, 0.25, 0.2, 0.1, 0.05, 0.05]  # More rain
        
        weather_conditions = list(self.weather_conditions.keys())
        return np.random.choice(weather_conditions, p=weather_probs)
    
    def _check_special_events(self, timestamp: datetime, location_name: str) -> Tuple[str, float]:
        """Check for special events that impact demand"""
        # 5% chance of special event
        if np.random.random() < 0.05:
            event_name = np.random.choice(list(self.special_events.keys()))
            event_data = self.special_events[event_name]
            
            if location_name in event_data['locations']:
                return event_name, event_data['multiplier']
        
        return None, 1.0
    
    def _generate_trip_distance(self, location_data: Dict) -> float:
        """Generate realistic trip distance"""
        if location_data['airport']:
            # Airport trips tend to be longer
            return max(0.5, np.random.exponential(12))
        else:
            # City trips shorter on average
            return max(0.2, np.random.exponential(4.5))
    
    def _calculate_fare(self, distance: float, demand: int) -> float:
        """Calculate fare based on Chicago pricing structure"""
        base_fare = 3.25  # Chicago base fare
        per_mile = 2.05   # Per mile rate
        
        # Surge pricing based on demand
        if demand > 40:
            surge_multiplier = 1.8
        elif demand > 30:
            surge_multiplier = 1.4
        elif demand > 20:
            surge_multiplier = 1.2
        else:
            surge_multiplier = 1.0
        
        fare = (base_fare + distance * per_mile) * surge_multiplier
        return round(fare, 2)
    
    def _get_season(self, timestamp: datetime) -> str:
        """Determine season from timestamp"""
        month = timestamp.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _calculate_distance_from_loop(self, lat: float, lon: float) -> float:
        """Calculate distance from Chicago Loop (downtown)"""
        loop_lat, loop_lon = 41.8781, -87.6298
        return round(np.sqrt((lat - loop_lat)**2 + (lon - loop_lon)**2), 4)
    
    def _get_temperature_range(self, timestamp: datetime, weather: str) -> str:
        """Estimate temperature range based on season and weather"""
        month = timestamp.month
        
        if month in [12, 1, 2]:  # Winter
            if weather == 'snow':
                return 'very_cold'  # < 20°F
            else:
                return 'cold'  # 20-40°F
        elif month in [6, 7, 8]:  # Summer
            return 'hot'  # > 75°F
        else:  # Spring/Fall
            return 'moderate'  # 40-75°F
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for ML model training"""
        
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Time-based features
        df['minutes_since_midnight'] = df['hour'] * 60 + df['timestamp'].dt.minute
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Lag features (previous hour demand - simplified)
        df = df.sort_values('timestamp')
        df['demand_lag_1h'] = df.groupby('location_name')['demand'].shift(1)
        df['demand_lag_24h'] = df.groupby('location_name')['demand'].shift(24)
        df['demand_lag_168h'] = df.groupby('location_name')['demand'].shift(168)  # 1 week
        
        # Fill NaN lag features with mean
        lag_cols = ['demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h']
        for col in lag_cols:
            df[col] = df[col].fillna(df[col].mean())
        
        # Moving averages
        df['demand_ma_3h'] = df.groupby('location_name')['demand'].rolling(3, min_periods=1).mean().values
        df['demand_ma_24h'] = df.groupby('location_name')['demand'].rolling(24, min_periods=1).mean().values
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = 'chicago_rides_realistic.csv'):
        """Save dataset to CSV with metadata"""
        filepath = f"/Users/aswin/time_series_forecasting/data/{filename}"
        df.to_csv(filepath, index=False)
        
        # Save metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'total_records': len(df),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'locations': df['location_name'].unique().tolist(),
            'weather_conditions': df['weather_condition'].unique().tolist(),
            'features': {
                'temporal': ['hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour'],
                'spatial': ['pickup_latitude', 'pickup_longitude', 'distance_from_loop'],
                'contextual': ['weather_condition', 'special_event', 'season'],
                'cyclical': ['hour_sin', 'hour_cos', 'day_sin', 'day_cos'],
                'derived': ['demand_lag_1h', 'demand_ma_3h']
            },
            'statistics': {
                'avg_demand': df['demand'].mean(),
                'max_demand': df['demand'].max(),
                'min_demand': df['demand'].min(),
                'std_demand': df['demand'].std()
            }
        }
        
        metadata_file = filepath.replace('.csv', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"💾 Dataset saved to: {filepath}")
        logger.info(f"📋 Metadata saved to: {metadata_file}")
        
        return filepath

def main():
    """Generate the complete Chicago dataset"""
    generator = ChicagoTransportationDataGenerator()
    
    # Generate 120,000 records for comprehensive training
    df = generator.generate_realistic_dataset(num_records=120000)
    
    # Save dataset
    filepath = generator.save_dataset(df)
    
    print(f"\n🎉 Chicago Transportation Dataset Generated Successfully!")
    print(f"📊 Dataset Shape: {df.shape}")
    print(f"💾 Saved to: {filepath}")
    print(f"🎯 Ready for ML model training!")

if __name__ == "__main__":
    main()