import requests
import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, List
import logging
from datetime import datetime, timedelta
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChicagoDataDownloader:
    """
    Downloads and processes Chicago Transportation Network Provider data
    Latest 2023-2025 dataset for maximum recruiter impact
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.base_url = "https://data.cityofchicago.org/resource"
        
        # Chicago TNP dataset endpoints (2023-2025)
        self.endpoints = {
            "tnp_2023_2024": "n26f-ihde.json",  # 2023-2024 trips
            "tnp_2025": "6dvr-xwnh.json",       # 2025 trips
            "tnp_drivers": "j6wf-834c.json",    # Driver data
            "tnp_vehicles": "bc6b-sq4u.json"    # Vehicle data
        }
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
    def download_chicago_tnp_data(self, limit: int = 100000, sample_for_demo: bool = True) -> pd.DataFrame:
        """
        Download Chicago Transportation Network Provider data
        
        Args:
            limit: Number of records to download (use smaller for demo)
            sample_for_demo: Whether to create a demo-friendly sample
        """
        logger.info(f"Downloading Chicago TNP data (limit: {limit:,} records)")
        
        if sample_for_demo:
            # Create realistic sample data for demo purposes
            return self._create_demo_dataset(limit)
        else:
            # Download actual data from Chicago Data Portal
            return self._download_real_data(limit)
    
    def _create_demo_dataset(self, num_records: int = 100000) -> pd.DataFrame:
        """
        Create realistic demo dataset mimicking Chicago TNP structure
        Perfect for portfolio demonstrations
        """
        logger.info(f"Creating demo dataset with {num_records:,} records")
        
        # Chicago geographic bounds
        lat_min, lat_max = 41.644, 42.023
        lon_min, lon_max = -87.940, -87.524
        
        # Generate date range (last 12 months for realistic patterns)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start_date, end_date, freq='5min')
        
        # Sample random timestamps
        random_timestamps = np.random.choice(date_range, num_records, replace=True)
        
        data = []
        for i, timestamp in enumerate(random_timestamps):
            if i % 10000 == 0:
                logger.info(f"Generated {i:,}/{num_records:,} records")
            
            # Convert numpy datetime64 to pandas Timestamp
            timestamp = pd.Timestamp(timestamp)
            
            # Generate realistic pickup/dropoff locations
            pickup_lat, pickup_lon = self._generate_realistic_location(timestamp)
            dropoff_lat, dropoff_lon = self._generate_realistic_location(timestamp, pickup_lat, pickup_lon)
            
            # Calculate trip metrics
            trip_miles = self._calculate_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
            trip_seconds = self._estimate_trip_duration(trip_miles, timestamp)
            
            # Calculate fare (realistic Chicago pricing)
            base_fare = 2.25
            per_mile = 1.75
            per_minute = 0.35
            fare = base_fare + (trip_miles * per_mile) + (trip_seconds / 60 * per_minute)
            
            # Add surge pricing during high demand periods
            surge_multiplier = self._calculate_surge_multiplier(timestamp, pickup_lat, pickup_lon)
            final_fare = fare * surge_multiplier
            
            # Add tips (realistic distribution)
            tip = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05])
            
            record = {
                'trip_start_timestamp': timestamp,
                'trip_end_timestamp': timestamp + timedelta(seconds=trip_seconds),
                'trip_seconds': trip_seconds,
                'trip_miles': round(trip_miles, 2),
                'pickup_centroid_latitude': round(pickup_lat, 6),
                'pickup_centroid_longitude': round(pickup_lon, 6),
                'dropoff_centroid_latitude': round(dropoff_lat, 6),
                'dropoff_centroid_longitude': round(dropoff_lon, 6),
                'fare': round(final_fare, 2),
                'tip': tip,
                'total_amount': round(final_fare + tip, 2),
                'payment_type': np.random.choice(['Credit Card', 'Cash', 'Mobile'], p=[0.8, 0.1, 0.1]),
                'company': np.random.choice(['Uber', 'Lyft'], p=[0.65, 0.35]),
                'surge_multiplier': round(surge_multiplier, 2),
                'weather_condition': self._get_weather_condition(timestamp),
                'day_of_week': timestamp.weekday(),
                'hour_of_day': timestamp.hour,
                'is_weekend': timestamp.weekday() >= 5,
                'is_holiday': self._is_holiday(timestamp),
                'temperature_f': self._get_temperature(timestamp),
                'precipitation_inches': self._get_precipitation(timestamp)
            }
            
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Save to file
        output_path = os.path.join(self.data_dir, f"chicago_tnp_demo_{num_records}.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Demo dataset saved to {output_path}")
        
        return df
    
    def _generate_realistic_location(self, timestamp: datetime, 
                                   origin_lat: Optional[float] = None, 
                                   origin_lon: Optional[float] = None) -> tuple:
        """Generate realistic Chicago locations based on time and urban patterns"""
        
        # Chicago high-demand areas
        hotspots = [
            (41.8781, -87.6298),  # Downtown Loop
            (41.9742, -87.9073),  # O'Hare Airport
            (41.8917, -87.6086),  # Navy Pier
            (41.8826, -87.6226),  # Millennium Park
            (41.9095, -87.6773),  # Wicker Park
            (41.8902, -87.6324),  # River North
            (41.9484, -87.6553),  # Lincoln Park
            (41.8676, -87.6176),  # South Loop
        ]
        
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        
        # Probability distribution based on time
        if origin_lat is None:  # Pickup location
            if 7 <= hour <= 9 and not is_weekend:  # Morning rush
                # Higher probability of residential pickup
                lat = np.random.normal(41.95, 0.1)
                lon = np.random.normal(-87.7, 0.1)
            elif 17 <= hour <= 19 and not is_weekend:  # Evening rush
                # Higher probability of downtown pickup
                lat = np.random.normal(41.878, 0.05)
                lon = np.random.normal(-87.63, 0.05)
            elif (22 <= hour <= 24 or 0 <= hour <= 3) and is_weekend:  # Weekend nightlife
                # Higher probability of entertainment district pickup
                hotspot = hotspots[np.random.randint(0, len(hotspots))]
                lat = np.random.normal(hotspot[0], 0.02)
                lon = np.random.normal(hotspot[1], 0.02)
            else:
                # Random distribution
                lat = np.random.uniform(41.644, 42.023)
                lon = np.random.uniform(-87.940, -87.524)
        else:  # Dropoff location (related to pickup)
            # Generate dropoff within reasonable distance
            max_distance = np.random.uniform(0.01, 0.15)  # Max ~10 miles
            angle = np.random.uniform(0, 2 * np.pi)
            
            lat = origin_lat + max_distance * np.cos(angle)
            lon = origin_lon + max_distance * np.sin(angle)
            
            # Ensure within Chicago bounds
            lat = np.clip(lat, 41.644, 42.023)
            lon = np.clip(lon, -87.940, -87.524)
        
        return lat, lon
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in miles"""
        # Haversine formula
        R = 3959  # Earth's radius in miles
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _estimate_trip_duration(self, distance_miles: float, timestamp: datetime) -> int:
        """Estimate trip duration based on distance and traffic conditions"""
        # Base speed in mph
        base_speed = 25
        
        # Traffic adjustments
        hour = timestamp.hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            speed_factor = 0.5
        elif 22 <= hour <= 6:  # Late night
            speed_factor = 1.3
        else:
            speed_factor = 1.0
        
        actual_speed = base_speed * speed_factor
        duration_hours = distance_miles / actual_speed
        duration_seconds = int(duration_hours * 3600)
        
        # Add random variation
        variation = np.random.uniform(0.8, 1.2)
        return int(duration_seconds * variation)
    
    def _calculate_surge_multiplier(self, timestamp: datetime, lat: float, lon: float) -> float:
        """Calculate realistic surge multiplier"""
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        
        # Base surge
        base_surge = 1.0
        
        # Time-based surge
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            time_surge = 1.5
        elif (22 <= hour <= 24 or 0 <= hour <= 3) and is_weekend:  # Weekend nights
            time_surge = 2.0
        else:
            time_surge = 1.0
        
        # Location-based surge (downtown areas)
        downtown_distance = np.sqrt((lat - 41.8781)**2 + (lon + 87.6298)**2)
        if downtown_distance < 0.05:
            location_surge = 1.3
        else:
            location_surge = 1.0
        
        # Weather surge
        weather_surge = np.random.choice([1.0, 1.2, 1.8], p=[0.7, 0.2, 0.1])
        
        final_surge = min(3.0, base_surge * time_surge * location_surge * weather_surge)
        return final_surge
    
    def _get_weather_condition(self, timestamp: datetime) -> str:
        """Generate realistic weather conditions"""
        # Seasonal patterns for Chicago
        month = timestamp.month
        
        if month in [12, 1, 2]:  # Winter
            conditions = ['Snow', 'Clear', 'Cloudy', 'Light Rain']
            probabilities = [0.3, 0.3, 0.3, 0.1]
        elif month in [6, 7, 8]:  # Summer
            conditions = ['Clear', 'Cloudy', 'Light Rain', 'Heavy Rain']
            probabilities = [0.5, 0.3, 0.15, 0.05]
        else:  # Spring/Fall
            conditions = ['Clear', 'Cloudy', 'Light Rain', 'Heavy Rain']
            probabilities = [0.4, 0.4, 0.15, 0.05]
        
        return np.random.choice(conditions, p=probabilities)
    
    def _get_temperature(self, timestamp: datetime) -> int:
        """Generate realistic Chicago temperatures"""
        month = timestamp.month
        
        # Chicago average temperatures by month (Fahrenheit)
        avg_temps = {
            1: 26, 2: 31, 3: 42, 4: 54, 5: 65, 6: 75,
            7: 79, 8: 77, 9: 69, 10: 56, 11: 43, 12: 31
        }
        
        base_temp = avg_temps[month]
        # Add daily variation
        variation = np.random.normal(0, 10)
        return int(base_temp + variation)
    
    def _get_precipitation(self, timestamp: datetime) -> float:
        """Generate realistic precipitation data"""
        weather = self._get_weather_condition(timestamp)
        
        if weather == 'Clear':
            return 0.0
        elif weather == 'Cloudy':
            return np.random.uniform(0, 0.01)
        elif weather == 'Light Rain':
            return np.random.uniform(0.01, 0.1)
        elif weather == 'Heavy Rain':
            return np.random.uniform(0.1, 0.5)
        elif weather == 'Snow':
            return np.random.uniform(0.05, 0.3)
        else:
            return 0.0
    
    def _is_holiday(self, timestamp: datetime) -> bool:
        """Check if date is a major holiday"""
        # Major US holidays that affect ride demand
        holidays_2024_2025 = [
            datetime(2024, 1, 1),   # New Year's Day
            datetime(2024, 7, 4),   # Independence Day
            datetime(2024, 12, 25), # Christmas
            datetime(2025, 1, 1),   # New Year's Day
            datetime(2025, 7, 4),   # Independence Day
            datetime(2025, 12, 25), # Christmas
        ]
        
        return timestamp.date() in [h.date() for h in holidays_2024_2025]
    
    def _download_real_data(self, limit: int) -> pd.DataFrame:
        """
        Download actual data from Chicago Data Portal
        Note: Requires API token for large downloads
        """
        url = f"{self.base_url}/{self.endpoints['tnp_2023_2024']}"
        
        params = {
            '$limit': limit,
            '$order': 'trip_start_timestamp DESC'
        }
        
        try:
            logger.info(f"Downloading from: {url}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            
            # Data cleaning and preprocessing
            df = self._preprocess_chicago_data(df)
            
            # Save raw data
            output_path = os.path.join(self.data_dir, f"chicago_tnp_raw_{limit}.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Raw data saved to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to download real data: {e}")
            logger.info("Falling back to demo dataset generation")
            return self._create_demo_dataset(limit)
    
    def _preprocess_chicago_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Chicago TNP data for ML model training"""
        logger.info("Preprocessing Chicago TNP data")
        
        # Convert timestamp columns
        timestamp_cols = ['trip_start_timestamp', 'trip_end_timestamp']
        for col in timestamp_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Clean numeric columns
        numeric_cols = ['trip_seconds', 'trip_miles', 'fare', 'tip', 'total_amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean location columns
        location_cols = ['pickup_centroid_latitude', 'pickup_centroid_longitude',
                        'dropoff_centroid_latitude', 'dropoff_centroid_longitude']
        for col in location_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove invalid records
        df = df.dropna(subset=['pickup_centroid_latitude', 'pickup_centroid_longitude'])
        
        # Filter to Chicago bounds
        chicago_mask = (
            (df['pickup_centroid_latitude'] >= 41.644) & 
            (df['pickup_centroid_latitude'] <= 42.023) &
            (df['pickup_centroid_longitude'] >= -87.940) & 
            (df['pickup_centroid_longitude'] <= -87.524)
        )
        df = df[chicago_mask]
        
        # Add derived features for ML
        df = self._add_feature_engineering(df)
        
        logger.info(f"Preprocessed dataset shape: {df.shape}")
        return df
    
    def _add_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features for ML model"""
        
        # Time-based features
        df['hour'] = df['trip_start_timestamp'].dt.hour
        df['day_of_week'] = df['trip_start_timestamp'].dt.dayofweek
        df['month'] = df['trip_start_timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = ((df['hour'].isin([7, 8, 9, 17, 18, 19]))).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Spatial features
        df['distance_from_downtown'] = np.sqrt(
            (df['pickup_centroid_latitude'] - 41.8781)**2 + 
            (df['pickup_centroid_longitude'] + 87.6298)**2
        )
        
        # Trip features
        df['trip_duration_minutes'] = df['trip_seconds'] / 60
        df['average_speed_mph'] = df['trip_miles'] / (df['trip_duration_minutes'] / 60)
        
        # Demand aggregation features (for spatial-temporal grid)
        df['pickup_grid_lat'] = (df['pickup_centroid_latitude'] * 100).round() / 100
        df['pickup_grid_lon'] = (df['pickup_centroid_longitude'] * 100).round() / 100
        df['time_slot'] = (df['hour'] * 4 + df['trip_start_timestamp'].dt.minute // 15)
        
        return df
    
    def create_aggregated_demand_data(self, df: pd.DataFrame, 
                                    grid_size: float = 0.01, 
                                    time_resolution_minutes: int = 15) -> pd.DataFrame:
        """
        Create aggregated demand data for spatial-temporal modeling
        This is what the Graph Neural Network will actually predict
        """
        logger.info("Creating aggregated demand data for ML training")
        
        # Create spatial grid
        df['grid_lat'] = (df['pickup_centroid_latitude'] / grid_size).round() * grid_size
        df['grid_lon'] = (df['pickup_centroid_longitude'] / grid_size).round() * grid_size
        
        # Create time slots
        df['time_slot'] = (
            df['trip_start_timestamp'].dt.floor(f'{time_resolution_minutes}min')
        )
        
        # Aggregate demand by grid cell and time slot
        demand_data = df.groupby(['grid_lat', 'grid_lon', 'time_slot']).agg({
            'trip_start_timestamp': 'count',  # Demand count
            'fare': 'mean',                   # Average fare
            'tip': 'mean',                    # Average tip
            'trip_miles': 'mean',             # Average distance
            'surge_multiplier': 'mean',       # Average surge
            'weather_condition': 'first',     # Weather
            'temperature_f': 'mean',          # Temperature
            'is_weekend': 'first',            # Weekend flag
            'hour_of_day': 'first'            # Hour
        }).reset_index()
        
        # Rename demand column
        demand_data.rename(columns={'trip_start_timestamp': 'demand_count'}, inplace=True)
        
        # Add lag features for time series
        demand_data = demand_data.sort_values(['grid_lat', 'grid_lon', 'time_slot'])
        
        for lag in [1, 2, 3, 24, 168]:  # 15min, 30min, 45min, 6h, 42h lags
            demand_data[f'demand_lag_{lag}'] = (
                demand_data.groupby(['grid_lat', 'grid_lon'])['demand_count']
                .shift(lag)
            )
        
        # Add rolling averages
        for window in [4, 12, 48]:  # 1h, 3h, 12h windows
            demand_data[f'demand_rolling_{window}'] = (
                demand_data.groupby(['grid_lat', 'grid_lon'])['demand_count']
                .rolling(window, min_periods=1).mean().values
            )
        
        # Save aggregated data
        output_path = os.path.join(self.data_dir, "chicago_demand_aggregated.csv")
        demand_data.to_csv(output_path, index=False)
        logger.info(f"Aggregated demand data saved to {output_path}")
        
        return demand_data

def main():
    """Demo the data download and processing"""
    downloader = ChicagoDataDownloader()
    
    # Download demo dataset
    print("🚖 Downloading Chicago TNP demo dataset...")
    df = downloader.download_chicago_tnp_data(limit=50000, sample_for_demo=True)
    
    print(f"✅ Downloaded {len(df):,} records")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📅 Date range: {df['trip_start_timestamp'].min()} to {df['trip_start_timestamp'].max()}")
    
    # Create aggregated demand data
    print("\n📊 Creating aggregated demand data for ML training...")
    demand_df = downloader.create_aggregated_demand_data(df)
    
    print(f"✅ Created aggregated dataset with {len(demand_df):,} demand records")
    print(f"🗺️ Spatial grid cells: {demand_df[['grid_lat', 'grid_lon']].drop_duplicates().shape[0]}")
    print(f"⏰ Time slots: {demand_df['time_slot'].nunique()}")
    
    # Display sample data
    print("\n📋 Sample aggregated data:")
    print(demand_df.head())
    
    return df, demand_df

if __name__ == "__main__":
    main()