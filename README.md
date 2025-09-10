# 🏙️ Chicago ML Demand Predictor

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-orange.svg)](https://pytorch.org)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io)
[![Weather API](https://img.shields.io/badge/Weather-OpenMeteo-lightblue.svg)](https://open-meteo.com)

A sophisticated machine learning application for predicting ride-sharing demand in Chicago using real-time weather data and neural networks. Built with Streamlit and powered by a custom-trained MassiveScaleDemandLSTM model.

![Chicago Skyline](images/winter_chicago.jpg)

> **🎯 For Recruiters**: [**Live Demo**](your-deployed-app-url) | [**Model Documentation**](exp.md) | [**Technical Deep-Dive**](detailed_explanation.md)

## ✨ Features

### 🧠 Advanced Machine Learning
- **MassiveScaleDemandLSTM**: Custom neural network with 2.1M+ parameters
- **Bidirectional LSTM + Multi-Head Attention**: Advanced architecture for temporal pattern recognition
- **19 Engineered Features**: Comprehensive feature set including temporal, spatial, and weather data
- **Real Chicago Data**: Trained on 200K-300K actual Chicago transportation records

### 🌤️ Real-Time Weather Integration
- **Live Weather API**: Real-time weather data from Open-Meteo API
- **Location-Specific Forecasts**: Weather data for specific Chicago coordinates
- **24-Hour Weather Timeline**: Comprehensive weather forecasting
- **Weather Impact Analysis**: Model predictions across different weather conditions

### 🎨 Beautiful UI/UX
- **Chicago Winter Theme**: Stunning Chicago skyline background
- **Light/Dark Mode**: Automatic theme detection with manual toggle
- **Responsive Design**: Optimized for desktop, tablet, and mobile
- **Modern Animations**: Smooth transitions and hover effects
- **Glass Morphism**: Advanced backdrop blur effects

### 📊 Interactive Visualizations
- **Chicago Heatmap**: Business district demand mapping
- **24-Hour Timeline**: Demand forecasting with confidence bands
- **Weather Impact Charts**: Real model predictions across weather conditions
- **Comparative Analysis**: Multi-location weather comparison

### 🎯 Business Intelligence
- **Demand Predictions**: Rides per hour for specific locations and times
- **Confidence Scoring**: Model-based prediction confidence
- **Business Recommendations**: Surge pricing and driver positioning advice
- **Location Context**: Business district categorization and insights

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/chicago-ml-demand-predictor.git
   cd chicago-ml-demand-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run chicago_ml_demand_predictor.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 📋 Requirements

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
plotly>=5.15.0
folium>=0.14.0
streamlit-folium>=0.13.0
torch>=1.11.0
scikit-learn>=1.1.0
requests>=2.28.0
joblib>=1.2.0
```

## 🏗️ Architecture

### Model Architecture
- **Input Layer**: 19 features → 256 dimensions with BatchNorm + ReLU + Dropout
- **LSTM Layers**: 2-layer Bidirectional LSTM (256 hidden units per direction)
- **Attention Layer**: 4-head Multi-Head Attention for temporal pattern capture
- **Output Layers**: 512 → 256 → 128 → 64 → 1 with ReLU activations
- **Final Activation**: ReLU for non-negative demand predictions

### Feature Engineering (19 Features)
#### Temporal Features (8)
- Hour of day, day of week, month, weekend flag
- Rush hour flag, business hours flag
- Cyclical encoding (hour_sin, hour_cos)

#### Spatial Features (4)
- Pickup latitude/longitude
- Distance from Chicago Loop center
- Downtown area flag

#### Historical Features (3)
- Demand lag (1 period, 8 periods)
- 3-period moving average

#### Weather Features (4)
- One-hot encoded weather conditions
- Clear, cloudy, rain, snow

### Data Pipeline
1. **Raw Data**: Chicago transportation API records
2. **Feature Engineering**: Extract temporal, spatial, contextual features
3. **Aggregation**: 3-hour time windows by location grid
4. **Lag Features**: Historical demand patterns
5. **Weather Integration**: Real-time weather conditions
6. **Scaling**: StandardScaler normalization
7. **Training**: 70% train, 15% validation, 15% test (temporal splits)

## 📁 Project Structure

```
chicago-ml-demand-predictor/
├── 📱 chicago_ml_demand_predictor.py    # Main Streamlit application
├── 🖼️ images/
│   └── winter_chicago.jpg               # Chicago skyline background
├── 🧠 src/
│   ├── model_integration/
│   │   └── checkpoint_loader.py         # Model loading and prediction
│   └── models/
│       └── real_demand_model.py         # Neural network architecture
├── 🔧 checkpoints/
│   ├── latest_checkpoint.pt             # Trained model weights
│   ├── feature_scaler.pkl              # Feature normalization
│   └── train.py                        # Training script
├── 📊 models/
│   └── model_metadata.json             # Model performance metrics
├── 📝 exp.md                           # Detailed model explanation
├── 📚 docs/                            # Additional documentation
├── 🐳 Dockerfile                       # Docker configuration
├── 📦 requirements.txt                 # Python dependencies
└── 📖 README.md                        # This file
```

## 🎯 Usage

### Basic Prediction
1. **Select Location**: Choose from 15 Chicago business districts
2. **Set Date & Time**: Pick prediction timestamp
3. **Get Prediction**: Real-time weather automatically fetched
4. **Analyze Results**: View demand prediction with confidence score

### Advanced Features
- **Weather Comparison**: See how different weather conditions affect demand
- **Timeline Analysis**: 24-hour demand forecasting with patterns
- **Business Intelligence**: Surge pricing and driver positioning recommendations
- **Location Comparison**: Compare demand across Chicago districts

### API Integration
The app automatically fetches:
- Real-time weather for specific coordinates
- 24-hour weather forecasts
- Location-specific temperature data
- Business district mapping

## 🎨 Customization

### Themes
- **Light Mode**: Clean, professional interface
- **Dark Mode**: Modern dark theme with blue accents
- **Auto-Detection**: Respects system preference
- **Manual Toggle**: Switch themes with top-right button

### Background
The Chicago winter skyline background can be customized by replacing `images/winter_chicago.jpg` with your preferred image.

## 📊 Model Performance

### Training Metrics
- **Validation Loss**: ~1.19 (MSE)
- **Training Epochs**: 18 with early stopping
- **Architecture**: Bidirectional LSTM + Multi-Head Attention
- **Parameters**: 2.1M trainable parameters

### Prediction Accuracy
- **R² Score**: Available in model metadata
- **Confidence Calculation**: Based on model training performance
- **Feature Importance**: 19 engineered features with temporal focus

## 🌐 Deployment

### Local Development
```bash
streamlit run chicago_ml_demand_predictor.py
```

### Docker Deployment
```bash
docker build -t chicago-ml-predictor .
docker run -p 8501:8501 chicago-ml-predictor
```

### Cloud Deployment
Compatible with:
- Streamlit Cloud
- Heroku
- AWS EC2
- Google Cloud Run
- Azure Container Instances

## 🔧 Configuration

### Environment Variables
- `WEATHER_API_TIMEOUT`: API request timeout (default: 10s)
- `MODEL_PATH`: Custom model checkpoint path
- `CACHE_TTL`: Weather data cache duration (default: 900s)

### Model Configuration
Models are loaded from the `checkpoints/` directory:
- `latest_checkpoint.pt`: Main model weights
- `feature_scaler.pkl`: Feature normalization
- `model_metadata.json`: Performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/chicago-ml-demand-predictor.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/

# Start development server
streamlit run chicago_ml_demand_predictor.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Chicago Open Data**: Transportation and geographic data
- **Open-Meteo API**: Real-time weather data
- **Streamlit**: Web application framework
- **PyTorch**: Deep learning framework
- **Plotly**: Interactive visualizations

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/chicago-ml-demand-predictor/issues)
- **Documentation**: [Model Documentation](exp.md)
- **Email**: your.email@example.com

## 🚨 Disclaimer

This application is for demonstration and educational purposes. Actual ride-sharing demand depends on many factors not captured in this model. Use predictions as estimates only.

---

**Built with ❤️ in Chicago** | **Powered by Neural Networks** | **Enhanced with Real-Time Data**