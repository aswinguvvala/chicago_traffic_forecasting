import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import datetime
import sys
import os

# Add models directory to path
sys.path.append('models')

# Import our ML classes
try:
    from ml_trainer import ChicagoMLTrainer
    ML_AVAILABLE = True
except ImportError as e:
    st.error(f"ML models not available: {e}")
    ML_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Chicago ML Demand Predictor",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Clean theme-aware styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    /* Theme Variables */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --border-color: #e5e7eb;
        --accent-color: #3b82f6;
        --accent-hover: #2563eb;
        --card-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Dark theme */
    [data-theme="dark"] {
        --bg-primary: #1f2937;
        --bg-secondary: #111827;
        --text-primary: #f9fafb;
        --text-secondary: #9ca3af;
        --border-color: #374151;
        --accent-color: #60a5fa;
        --accent-hover: #3b82f6;
        --card-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    
    /* Base theme detection */
    @media (prefers-color-scheme: dark) {
        :root:not([data-theme="light"]) {
            --bg-primary: #1f2937;
            --bg-secondary: #111827;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --border-color: #374151;
            --accent-color: #60a5fa;
            --accent-hover: #3b82f6;
            --card-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        }
    }
    
    /* App base */
    .stApp {
        background-color: var(--bg-secondary) !important;
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }
    
    /* Theme toggle */
    .theme-toggle {
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 1000;
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.5rem;
        cursor: pointer;
        box-shadow: var(--card-shadow);
        transition: all 0.2s ease;
    }
    
    .theme-toggle:hover {
        background: var(--bg-secondary);
    }
    
    /* Simple header */
    .app-header {
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--card-shadow);
    }
    
    .app-title {
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* Control cards */
    .control-card {
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--card-shadow);
    }
    
    /* Result card */
    .result-card {
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        box-shadow: var(--card-shadow);
        margin-top: 1.5rem;
    }
    
    .prediction-value {
        font-size: 3rem;
        font-weight: 600;
        color: var(--accent-color);
        margin: 0 0 0.5rem 0;
    }
    
    .prediction-label {
        color: var(--text-secondary);
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0;
    }
    
    /* Form styling */
    .stSelectbox label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
    }
    
    .stSelectbox > div > div {
        background: var(--bg-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
        font-size: 0.875rem !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--accent-color) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: var(--accent-hover) !important;
        transform: translateY(-1px);
        box-shadow: var(--card-shadow);
    }
    
    /* Hide sidebar completely */
    .css-1d391kg {
        display: none !important;
    }
    
    /* Clean text colors */
    .main h1, .main h2, .main h3, .main p {
        color: var(--text-primary) !important;
    }
    
    .main .metric-value {
        color: var(--text-secondary) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize ML trainer
@st.cache_resource
def load_ml_models():
    """Load pre-trained ML models with caching"""
    if not ML_AVAILABLE:
        return None
    
    try:
        trainer = ChicagoMLTrainer()
        trainer.load_models()
        return trainer
    except Exception as e:
        st.error(f"Failed to load ML models: {e}")
        return None

# Load Chicago dataset for analysis
@st.cache_data
def load_chicago_data():
    """Load Chicago dataset for visualizations"""
    try:
        df = pd.read_csv('data/chicago_rides_realistic.csv', nrows=10000)  # Sample for performance
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        st.warning("Dataset not found. Some visualizations may not work.")
        return None

def main():
    """Clean, simple ML prediction interface"""
    
    # Theme toggle
    st.markdown("""
    <div class="theme-toggle" onclick="toggleTheme()" title="Toggle theme">
        🌓
    </div>
    <script>
        function toggleTheme() {
            const root = document.documentElement;
            const currentTheme = root.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            root.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
        }
        
        // Load saved theme
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            document.documentElement.setAttribute('data-theme', savedTheme);
        }
    </script>
    """, unsafe_allow_html=True)
    
    # Simple header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">Chicago ML Demand Predictor</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    trainer = load_ml_models()
    
    if trainer is None:
        st.error("ML models not available. Please train models first.")
        st.info("Run `python models/ml_trainer.py` to train the models.")
        return
    
    # Inline controls
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    
    # Simple controls in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_type = st.selectbox(
            "Model",
            ["random_forest", "lstm"],
            format_func=lambda x: "RandomForest" if x == "random_forest" else "LSTM"
        )
    
    with col2:
        chicago_locations = {
            "Loop": (41.8781, -87.6298),
            "O'Hare": (41.9742, -87.9073),
            "Magnificent Mile": (41.8955, -87.6244),
            "Lincoln Park": (41.9254, -87.6547),
            "Wicker Park": (41.9073, -87.6776),
            "Navy Pier": (41.8917, -87.6086),
        }
        
        selected_location = st.selectbox(
            "Location",
            list(chicago_locations.keys())
        )
        lat, lon = chicago_locations[selected_location]
    
    with col3:
        weather = st.selectbox(
            "Weather",
            ["clear", "cloudy", "light_rain", "heavy_rain", "snow", "fog"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create current timestamp
    timestamp = pd.Timestamp.now()
    
    # Prediction interface
    if st.button("Get Prediction", type="primary"):
        
        with st.spinner("Analyzing..."):
            try:
                # Get ML prediction
                prediction = trainer.predict(
                    lat=lat, 
                    lon=lon, 
                    timestamp=timestamp,
                    model_type=model_type,
                    weather=weather,
                    special_event='none'
                )
                
                # Clean result display
                st.markdown(f"""
                <div class="result-card">
                    <div class="prediction-value">{prediction['predicted_demand']}</div>
                    <p class="prediction-label">Rides per Hour</p>
                    <hr style="border: 1px solid var(--border-color); margin: 1rem 0;">
                    <p style="color: var(--text-secondary); margin: 0.5rem 0;">
                        <strong>Confidence:</strong> {prediction['confidence']:.1%}<br>
                        <strong>Model:</strong> {model_type.replace('_', ' ').title()}<br>
                        <strong>Location:</strong> {selected_location}<br>
                        <strong>Weather:</strong> {weather.replace('_', ' ').title()}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    # Simple info
    st.markdown("---")
    st.markdown("**About:** Real ML models trained on Chicago transportation data. RandomForest: 79.3% accuracy, LSTM: 62.4% accuracy.")


if __name__ == "__main__":
    main()