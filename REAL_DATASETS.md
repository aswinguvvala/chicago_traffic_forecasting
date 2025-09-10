# 🚖 Real-World Transportation Datasets for ML Training

## **Overview**
This document contains curated real-world datasets that can be used to train demand forecasting models for ride-hailing services. All datasets listed here contain actual historical data rather than synthetic data.

---

## **🌟 Primary Recommended Datasets**

### **1. NYC Taxi & Limousine Commission (TLC) Trip Record Data**
- **Source**: [NYC Open Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Description**: Official NYC taxi trip records with pickup/dropoff locations, times, and fares
- **Size**: ~100M+ trips per year
- **Time Range**: 2009 - Present (monthly updates)
- **Format**: Parquet/CSV files
- **Key Features**:
  - Pickup/dropoff datetime
  - GPS coordinates (latitude/longitude)
  - Trip distance and duration
  - Fare amounts
  - Payment type
  - Passenger count

**Download URLs**:
```
# Yellow Taxi Data (most recent)
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet

# Green Taxi Data
https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-01.parquet

# For-Hire Vehicle (includes Uber/Lyft)
https://d37ci6vzurychx.cloudfront.net/trip-data/fhv_tripdata_2024-01.parquet
```

### **2. Chicago Transportation Network Providers (TNP) Data**
- **Source**: [Chicago Data Portal](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p)
- **Description**: Official Chicago ride-hailing trip data (Uber, Lyft, etc.)
- **Size**: ~30M+ trips per year
- **Time Range**: 2018 - Present
- **Format**: CSV/JSON via API
- **Key Features**:
  - Trip start/end timestamps
  - Pickup/dropoff community areas
  - Trip seconds and miles
  - Fare amount
  - Tips

**API Access**:
```python
# Chicago TNP Data API
base_url = "https://data.cityofchicago.org/resource/m6dm-c72p.json"
# Add filters: ?$where=trip_start_timestamp>'2024-01-01T00:00:00'
```

### **3. San Francisco Taxi Data**
- **Source**: [SF Open Data](https://data.sfgov.org/Transportation/Taxi-Trips/rqzk-sfat)
- **Description**: San Francisco taxi trip records
- **Size**: ~500K+ trips per month
- **Time Range**: 2012 - Present
- **Key Features**:
  - Pickup/dropoff coordinates
  - Trip start/end times
  - Fare details

---

## **🌧️ Weather Data Sources**

### **1. OpenWeatherMap Historical API**
- **Source**: [OpenWeatherMap](https://openweathermap.org/api/statistics-api)
- **Description**: Historical weather data for any city
- **Features**: Temperature, precipitation, wind, visibility
- **Cost**: Free tier available, paid for bulk historical data

### **2. NOAA Weather Data**
- **Source**: [NOAA Climate Data](https://www.ncdc.noaa.gov/data-access)
- **Description**: US government weather data
- **Features**: Comprehensive historical weather records
- **Cost**: Free

---

## **📊 Additional Data Sources**

### **1. Google Traffic Data**
- **Source**: Google Maps API
- **Description**: Real-time and historical traffic conditions
- **Cost**: Paid API

### **2. Events Data**
- **Eventbrite API**: Concerts, festivals, conferences
- **Sports APIs**: NBA, NFL, MLB game schedules
- **Meetup API**: Local events and gatherings

### **3. Economic Indicators**
- **Federal Reserve Economic Data (FRED)**: Employment, gas prices
- **Bureau of Labor Statistics**: Economic indicators by city

---

## **💾 Data Download Scripts**

### **NYC TLC Data Downloader**
```python
import pandas as pd
import requests
from datetime import datetime, timedelta

def download_nyc_taxi_data(year=2024, months=None):
    \"\"\"Download NYC taxi data for training\"\"\"
    
    if months is None:
        months = list(range(1, 13))  # All months
    
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    
    for month in months:
        # Yellow taxi data
        filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
        url = f"{base_url}{filename}"
        
        print(f"Downloading {filename}...")
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(f"data/raw/{filename}", "wb") as f:
                f.write(response.content)
            print(f"✅ Downloaded {filename}")
        else:
            print(f"❌ Failed to download {filename}")

# Usage
download_nyc_taxi_data(year=2024, months=[1, 2, 3])
```

### **Chicago TNP Data Fetcher**
```python
import requests
import pandas as pd
from datetime import datetime

def fetch_chicago_tnp_data(start_date="2024-01-01", limit=50000):
    \"\"\"Fetch Chicago TNP data via API\"\"\"
    
    base_url = "https://data.cityofchicago.org/resource/m6dm-c72p.json"
    
    params = {
        "$where": f"trip_start_timestamp > '{start_date}T00:00:00'",
        "$limit": limit,
        "$order": "trip_start_timestamp DESC"
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        return df
    else:
        raise Exception(f"API request failed: {response.status_code}")

# Usage
chicago_data = fetch_chicago_tnp_data(start_date="2024-01-01", limit=100000)
```

---

## **🎯 Data Preprocessing Requirements**

### **1. Temporal Features**
- Extract hour, day of week, month from timestamps
- Create cyclical encodings (sin/cos) for temporal features
- Identify rush hours, weekends, holidays

### **2. Spatial Features**
- Convert GPS coordinates to zone/grid system
- Calculate distances from key landmarks (airports, downtown, stadiums)
- Create spatial clusters for similar demand patterns

### **3. Demand Aggregation**
- Aggregate trip counts by time windows (15-min, 30-min, 1-hour)
- Create grid-based demand heatmaps
- Calculate demand density by area

### **4. External Data Integration**
- Join weather data by timestamp and location
- Merge event data for demand spikes
- Include traffic conditions if available

---

## **⚡ Google Colab Training Strategy**

### **1. Data Loading Strategy**
```python
# Load data in chunks to manage memory
chunk_size = 100000
data_chunks = pd.read_csv('large_dataset.csv', chunksize=chunk_size)

processed_data = []
for chunk in data_chunks:
    # Process each chunk
    processed_chunk = preprocess_chunk(chunk)
    processed_data.append(processed_chunk)

final_data = pd.concat(processed_data, ignore_index=True)
```

### **2. GPU Memory Management**
```python
import torch

# Clear GPU memory
torch.cuda.empty_cache()

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Gradient accumulation for large batches
accumulation_steps = 4
```

### **3. Checkpointing Strategy**
```python
# Save checkpoints every N epochs
checkpoint_interval = 5

# Include all necessary components in checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'scaler_state_dict': scaler.state_dict(),
    'feature_columns': feature_columns,
    'model_config': model_config,
    'training_stats': training_stats
}

torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')
```

---

## **📈 Expected Data Volumes**

| Dataset | Monthly Size | Annual Size | Rows/Month |
|---------|-------------|-------------|------------|
| NYC Taxi | ~8GB | ~100GB | ~8M trips |
| Chicago TNP | ~500MB | ~6GB | ~2M trips |
| Weather Data | ~10MB | ~120MB | ~50K records |

---

## **🚀 Next Steps**

1. **Choose Primary Dataset**: NYC TLC data recommended for largest volume
2. **Download Historical Data**: Get 12+ months for seasonal patterns
3. **Set Up Data Pipeline**: Automated preprocessing and feature engineering
4. **Train Initial Models**: Start with simple LSTM, progress to GNN-LSTM
5. **Validate Performance**: Use real hold-out test sets for honest evaluation

---

## **⚠️ Important Notes**

- **Data Privacy**: All datasets are aggregated and anonymized
- **Usage Rights**: Check license requirements for commercial use
- **Update Frequency**: NYC updates monthly, Chicago updates weekly
- **Data Quality**: Some missing values and outliers expected in real data
- **Computational Requirements**: Large datasets require significant compute resources

---

This document provides the foundation for training real ML models on actual transportation data rather than synthetic mock data.

