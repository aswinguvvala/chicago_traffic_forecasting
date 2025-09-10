# 🚀 Deployment Guide - Uber Demand Forecasting

## 📋 **Quick Start for Recruiters**

### **Option 1: Local Demo (Recommended for Interviews)**

```bash
# Clone repository
git clone https://github.com/yourusername/uber-demand-forecasting
cd uber-demand-forecasting

# Install dependencies
pip install -r requirements.txt

# Launch recruiter showcase
streamlit run app.py
```

**⏱️ Setup time: 2-3 minutes**  
**🎯 Best for: Live demos, technical interviews, portfolio reviews**

### **Option 2: Cloud Deployment (Production Ready)**

Deploy on **Streamlit Community Cloud**, **Heroku**, or **AWS** for permanent showcase.

---

## 🏗️ **Architecture Overview**

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌────────────────┐│
│  │ Streamlit       │ │ Recruiter       │ │ Interactive    ││
│  │ Dashboard       │ │ Showcase        │ │ Demo           ││
│  └─────────────────┘ └─────────────────┘ └────────────────┘│
└─────────────────────┬───────────────────────────────────────┘
                      │ RESTful API / Local Calls
┌─────────────────────▼───────────────────────────────────────┐
│                    Backend Layer                            │
│  ┌─────────────────┐ ┌─────────────────┐ ┌────────────────┐│
│  │ FastAPI         │ │ Business Logic  │ │ Data           ││
│  │ Server          │ │ & ROI Calc      │ │ Processing     ││
│  └─────────────────┘ └─────────────────┘ └────────────────┘│
└─────────────────────┬───────────────────────────────────────┘
                      │ Model Inference
┌─────────────────────▼───────────────────────────────────────┐
│                    ML Model Layer                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌────────────────┐│
│  │ Graph Neural    │ │ LSTM Network    │ │ Feature        ││
│  │ Network         │ │ (Temporal)      │ │ Engineering    ││
│  │ (Spatial)       │ │                 │ │ Pipeline       ││
│  └─────────────────┘ └─────────────────┘ └────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 **Local Development Setup**

### **Prerequisites**
- Python 3.8+ (3.9+ recommended)
- 8GB+ RAM (for full dataset processing)
- Modern web browser

### **Step 1: Environment Setup**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### **Step 2: Data Preparation**

```bash
# Generate demo dataset
python src/data_processing/data_downloader.py

# Train model (optional - demo uses pre-trained simulation)
python src/models/gnn_lstm_model.py
```

### **Step 3: Launch Applications**

**Main Portfolio App:**
```bash
streamlit run app.py
```

**API Server (optional):**
```bash
cd api && python main.py
```

**Individual Components:**
```bash
# Recruiter showcase only
streamlit run demo/recruiter_showcase.py

# Technical dashboard only  
streamlit run dashboard/uber_demand_dashboard.py

# EDA notebook
jupyter notebook notebooks/01_EDA_Business_Insights.ipynb
```

---

## ☁️ **Cloud Deployment Options**

### **Streamlit Community Cloud (Recommended for Portfolio)**

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add Uber demand forecasting portfolio"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect GitHub repository
   - Select `app.py` as main file
   - Deploy automatically

3. **Configuration**
   ```toml
   # .streamlit/config.toml
   [theme]
   primaryColor = "#667eea"
   backgroundColor = "#FFFFFF"
   secondaryBackgroundColor = "#f0f2f6"
   textColor = "#262730"
   font = "sans serif"
   ```

**⏱️ Deployment time: 5-10 minutes**  
**💰 Cost: Free**  
**🎯 Perfect for: Portfolio showcases, recruiter demos**

### **Heroku Deployment**

1. **Heroku Setup**
   ```bash
   # Install Heroku CLI
   heroku create uber-demand-forecasting-demo
   
   # Configure buildpacks
   heroku buildpacks:set heroku/python
   ```

2. **Configuration Files**
   
   **Procfile:**
   ```
   web: sh setup.sh && streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
   
   **setup.sh:**
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```

3. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

**⏱️ Deployment time: 10-15 minutes**  
**💰 Cost: Free tier available**  
**🎯 Best for: Professional portfolio with custom domain**

### **AWS Deployment (Advanced)**

**Using AWS Elastic Beanstalk:**

1. **Application Structure**
   ```
   application.py  # Renamed from app.py
   requirements.txt
   .ebextensions/
     01_streamlit.config
   ```

2. **EB Configuration**
   ```yaml
   # .ebextensions/01_streamlit.config
   option_settings:
     aws:elasticbeanstalk:container:python:
       WSGIPath: application.py
     aws:elasticbeanstalk:application:environment:
       STREAMLIT_SERVER_HEADLESS: true
   ```

3. **Deploy**
   ```bash
   eb init uber-demand-forecasting
   eb create production
   eb deploy
   ```

**⏱️ Deployment time: 15-20 minutes**  
**💰 Cost: Pay-as-you-go**  
**🎯 Best for: Production-grade deployments**

---

## 🎯 **Recruiter-Focused Deployment**

### **Quick Demo Setup**

For **immediate recruiter demos** during interviews:

```bash
# 1-minute setup
git clone https://github.com/yourusername/uber-demand-forecasting
cd uber-demand-forecasting
pip install streamlit pandas plotly numpy
streamlit run demo/recruiter_showcase.py
```

### **Portfolio Integration**

Add to your **existing portfolio website**:

```html
<!-- Embed Streamlit app -->
<iframe src="https://your-app.streamlit.app" 
        width="100%" height="800px" 
        frameborder="0">
</iframe>
```

### **Demo Presentation Mode**

**For live presentations:**

```bash
# Full screen mode
streamlit run app.py --server.headless=true --server.runOnSave=true
```

**Browser shortcuts:**
- `F11` - Full screen
- `Ctrl+Shift+I` - Hide browser interface
- `Ctrl+R` - Refresh demo

---

## 🔒 **Security & Performance**

### **Environment Variables**

```bash
# .env file
API_KEY=your-weather-api-key
MODEL_PATH=models/gnn_lstm_model.pt
DEBUG=False
STREAMLIT_SERVER_HEADLESS=true
```

### **Performance Optimization**

```python
# streamlit_config.py
import streamlit as st

# Cache data loading
@st.cache_data
def load_chicago_data():
    return pd.read_csv('data/chicago_tnp_demo.csv')

# Cache model predictions
@st.cache_data(ttl=300)  # 5-minute cache
def predict_demand(lat, lon, timestamp):
    return model.predict(lat, lon, timestamp)
```

### **Resource Management**

```toml
# .streamlit/config.toml
[server]
maxUploadSize = 50
maxMessageSize = 50

[browser]
gatherUsageStats = false
```

---

## 📊 **Monitoring & Analytics**

### **Streamlit Analytics**

```python
# Add to app.py
import streamlit as st

# Track usage for recruiters
if 'recruiter_visits' not in st.session_state:
    st.session_state.recruiter_visits = 0
    
st.session_state.recruiter_visits += 1
```

### **API Monitoring**

```python
# api/monitoring.py
from fastapi import FastAPI
import time

app = FastAPI()

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

**Issue 1: Memory errors with large dataset**
```bash
# Solution: Use sample data for demos
export DEMO_MODE=true
python src/data_processing/data_downloader.py
```

**Issue 2: Slow model loading**
```python
# Solution: Use cached predictions
@st.cache_resource
def load_model():
    return UberDemandPredictor()
```

**Issue 3: Port conflicts**
```bash
# Solution: Use different port
streamlit run app.py --server.port 8502
```

### **Performance Issues**

**Slow startup:**
- Reduce dataset size for demos
- Use cached model predictions
- Enable Streamlit caching

**Memory usage:**
- Limit concurrent users
- Use data sampling
- Implement lazy loading

---

## 🎯 **Recruiter Checklist**

Before sharing with recruiters:

- [ ] ✅ App loads within 30 seconds
- [ ] ✅ All interactive features work
- [ ] ✅ Business metrics display correctly
- [ ] ✅ ROI calculator functions properly
- [ ] ✅ Contact information is updated
- [ ] ✅ GitHub repository is public
- [ ] ✅ Demo video is accessible (optional)
- [ ] ✅ Mobile responsiveness tested

---

## 📞 **Support**

**For deployment issues:**
- 📧 Email: your.email@domain.com
- 💻 GitHub Issues: [Create Issue](https://github.com/yourusername/uber-demand-forecasting/issues)
- 📱 LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

**Demo scheduling:**
- 📅 Available for live demos
- 🎥 Recorded demo available
- ⚡ Quick setup for technical interviews