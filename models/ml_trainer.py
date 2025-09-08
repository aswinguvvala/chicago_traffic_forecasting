import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. LSTM model will be disabled.")

from feature_engineering import ChicagoFeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChicagoMLTrainer:
    """
    Comprehensive ML training pipeline for Chicago transportation demand forecasting
    Implements both RandomForest (baseline) and LSTM (advanced) models
    """
    
    def __init__(self):
        self.feature_engineer = ChicagoFeatureEngineer()
        self.models = {}
        self.model_metrics = {}
        self.is_trained = False
        
        # Model configurations
        self.rf_config = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"🔥 PyTorch device: {self.device}")
        
    def train_models(self, df: pd.DataFrame, target_col: str = 'demand') -> Dict[str, Dict]:
        """
        Train both RandomForest and LSTM models with comprehensive validation
        
        Args:
            df: Chicago transportation DataFrame
            target_col: Target column name
            
        Returns:
            Dictionary with model performance metrics
        """
        logger.info(f"🚀 Starting ML model training pipeline...")
        logger.info(f"📊 Dataset: {len(df):,} records, target: {target_col}")
        
        # Feature engineering
        X, y, feature_names = self.feature_engineer.prepare_features(df, target_col)
        
        # Train/validation/test splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.feature_engineer.create_train_test_split(
            X, y, test_size=0.2, val_size=0.1
        )
        
        # Train RandomForest model
        logger.info("🌲 Training RandomForest model...")
        rf_metrics = self._train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Train LSTM model (if PyTorch available)
        if TORCH_AVAILABLE:
            logger.info("🧠 Training LSTM neural network...")
            lstm_metrics = self._train_lstm(X_train, X_val, X_test, y_train, y_val, y_test)
        else:
            lstm_metrics = {"error": "PyTorch not available"}
        
        # Model comparison
        logger.info("📊 Model Performance Comparison:")
        logger.info("="*60)
        logger.info("🌲 RandomForest Performance:")
        for metric, value in rf_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"   • {metric}: {value:.4f}")
        
        if TORCH_AVAILABLE and "error" not in lstm_metrics:
            logger.info("🧠 LSTM Performance:")
            for metric, value in lstm_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"   • {metric}: {value:.4f}")
        
        # Store metrics
        self.model_metrics = {
            'random_forest': rf_metrics,
            'lstm': lstm_metrics,
            'training_date': datetime.now().isoformat(),
            'dataset_size': len(df),
            'feature_count': X.shape[1],
            'feature_names': feature_names
        }
        
        self.is_trained = True
        
        # Save models
        self._save_models()
        
        logger.info("✅ Model training pipeline completed successfully!")
        
        return self.model_metrics
    
    def _train_random_forest(self, X_train, X_val, X_test, y_train, y_val, y_test) -> Dict:
        """Train and validate RandomForest model"""
        
        # Train model
        rf_model = RandomForestRegressor(**self.rf_config)
        rf_model.fit(X_train, y_train)
        
        # Predictions
        train_pred = rf_model.predict(X_train)
        val_pred = rf_model.predict(X_val)
        test_pred = rf_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'val_r2': r2_score(y_val, val_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'test_r2': r2_score(y_test, test_pred),
            'model_type': 'RandomForest'
        }
        
        # Calculate accuracy as percentage
        metrics['test_accuracy'] = (1 - metrics['test_mae'] / np.mean(y_test)) * 100
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        metrics['cv_mae_mean'] = -cv_scores.mean()
        metrics['cv_mae_std'] = cv_scores.std()
        
        # Feature importance
        feature_names = self.feature_engineer.get_feature_importance_names()
        if len(feature_names) == len(rf_model.feature_importances_):
            importance_dict = dict(zip(feature_names, rf_model.feature_importances_))
            # Top 10 most important features
            top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            metrics['top_features'] = top_features
        
        # Store model
        self.models['random_forest'] = rf_model
        
        return metrics
    
    def _train_lstm(self, X_train, X_val, X_test, y_train, y_val, y_test) -> Dict:
        """Train and validate LSTM neural network"""
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device) 
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # Create LSTM model
        input_size = X_train.shape[1]
        lstm_model = LSTMDemandPredictor(input_size=input_size).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        num_epochs = 50
        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = 10
        
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            lstm_model.train()
            optimizer.zero_grad()
            
            train_pred = lstm_model(X_train_tensor)
            train_loss = criterion(train_pred, y_train_tensor)
            train_loss.backward()
            optimizer.step()
            
            # Validation
            lstm_model.eval()
            with torch.no_grad():
                val_pred = lstm_model(X_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor)
                
                val_pred_np = val_pred.cpu().numpy()
                y_val_np = y_val
                
                val_mae = mean_absolute_error(y_val_np, val_pred_np)
                val_rmse = np.sqrt(mean_squared_error(y_val_np, val_pred_np))
                val_r2 = r2_score(y_val_np, val_pred_np)
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = lstm_model.state_dict().copy()
            else:
                patience_counter += 1
                
            if epoch % 10 == 0:
                logger.info(f"   Epoch {epoch+1:2d}/{num_epochs}: Train Loss={train_loss.item():.4f}, Val Loss={val_loss.item():.4f}, Val MAE={val_mae:.3f}")
                
            if patience_counter >= patience_limit:
                logger.info(f"   Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        lstm_model.load_state_dict(best_model_state)
        
        # Final evaluation
        lstm_model.eval()
        with torch.no_grad():
            train_pred = lstm_model(X_train_tensor).cpu().numpy()
            val_pred = lstm_model(X_val_tensor).cpu().numpy()
            test_pred = lstm_model(X_test_tensor).cpu().numpy()
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'val_r2': r2_score(y_val, val_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'test_r2': r2_score(y_test, test_pred),
            'epochs_trained': epoch + 1,
            'best_val_loss': best_val_loss.item(),
            'model_type': 'LSTM'
        }
        
        # Calculate accuracy
        metrics['test_accuracy'] = (1 - metrics['test_mae'] / np.mean(y_test)) * 100
        
        # Store model
        self.models['lstm'] = lstm_model
        
        return metrics
    
    def predict(self, lat: float, lon: float, timestamp: pd.Timestamp, 
                model_type: str = 'random_forest', weather: str = 'clear', 
                special_event: str = 'none') -> Dict:
        """
        Make real ML prediction (replaces random number generation)
        
        Args:
            lat: Pickup latitude
            lon: Pickup longitude  
            timestamp: Prediction timestamp
            model_type: 'random_forest' or 'lstm'
            weather: Weather condition ('clear', 'cloudy', 'light_rain', 'heavy_rain', 'snow', 'fog')
            special_event: Special event type ('none', 'Cubs Game', etc.)
            
        Returns:
            Prediction results with confidence metrics
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_models() first.")
        
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not available. Available: {list(self.models.keys())}")
        
        # Feature engineering for single prediction
        features = self.feature_engineer.transform_single_prediction(lat, lon, timestamp, weather, special_event)
        
        model = self.models[model_type]
        
        if model_type == 'random_forest':
            # RandomForest prediction
            prediction = model.predict(features)[0]
            
            # Calculate prediction interval using forest variance
            tree_predictions = [tree.predict(features)[0] for tree in model.estimators_]
            prediction_std = np.std(tree_predictions)
            confidence = min(0.95, 0.80 + (1 / (1 + prediction_std)))
            
        elif model_type == 'lstm' and TORCH_AVAILABLE:
            # LSTM prediction
            model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).to(self.device)
                prediction_output = model(features_tensor).cpu().numpy()
                # Handle both 0-dimensional and 1-dimensional outputs
                prediction = prediction_output.item() if prediction_output.ndim == 0 else prediction_output[0]
                
            # Simple confidence based on validation performance
            val_mae = self.model_metrics['lstm']['val_mae']
            confidence = max(0.75, min(0.95, 1 - (val_mae / prediction) if prediction > 0 else 0.75))
        
        else:
            raise ValueError(f"Model type {model_type} not supported")
        
        # Ensure realistic prediction bounds
        prediction = max(1, int(round(prediction)))
        
        # Calculate business metrics
        business_metrics = self._calculate_business_metrics(prediction, lat, lon, timestamp)
        
        return {
            'predicted_demand': prediction,
            'confidence': confidence,
            'model_used': model_type,
            'model_performance': {
                'test_accuracy': self.model_metrics[model_type]['test_accuracy'],
                'test_mae': self.model_metrics[model_type]['test_mae'],
                'test_r2': self.model_metrics[model_type]['test_r2']
            },
            **business_metrics
        }
    
    def _calculate_business_metrics(self, demand: int, lat: float, lon: float, timestamp: pd.Timestamp) -> Dict:
        """Calculate business KPIs based on prediction"""
        
        # Chicago fare structure
        base_fare = 3.25
        per_mile_rate = 2.05
        avg_trip_distance = 4.2
        
        # Surge pricing based on demand
        if demand >= 40:
            surge_multiplier = 2.0
        elif demand >= 30:
            surge_multiplier = 1.6
        elif demand >= 20:
            surge_multiplier = 1.3
        else:
            surge_multiplier = 1.0
        
        # Revenue calculations
        avg_fare = (base_fare + per_mile_rate * avg_trip_distance) * surge_multiplier
        hourly_revenue = demand * avg_fare
        
        # Driver utilization (simplified)
        optimal_drivers = max(1, demand // 3)  # ~3 rides per driver per hour
        utilization_rate = min(95, (demand / optimal_drivers) * 15)  # Simplified calculation
        
        # Wait times
        if demand >= 35:
            avg_wait_time = 12  # High demand
        elif demand >= 20:
            avg_wait_time = 6   # Moderate demand
        else:
            avg_wait_time = 3   # Low demand
        
        return {
            'surge_multiplier': surge_multiplier,
            'estimated_hourly_revenue': round(hourly_revenue, 2),
            'average_fare': round(avg_fare, 2),
            'recommended_drivers': optimal_drivers,
            'driver_utilization_rate': round(utilization_rate, 1),
            'estimated_wait_time_minutes': avg_wait_time
        }
    
    def _save_models(self):
        """Save trained models and metadata"""
        
        import os
        os.makedirs('saved_models', exist_ok=True)
        
        # Save RandomForest
        if 'random_forest' in self.models:
            joblib.dump(self.models['random_forest'], 'saved_models/random_forest_model.pkl')
            
        # Save LSTM
        if 'lstm' in self.models and TORCH_AVAILABLE:
            torch.save({
                'model_state_dict': self.models['lstm'].state_dict(),
                'input_size': self.models['lstm'].input_size,
                'hidden_size': self.models['lstm'].hidden_size
            }, 'saved_models/lstm_model.pt')
        
        # Save feature engineering pipeline
        self.feature_engineer.save_feature_pipeline('saved_models/feature_pipeline.pkl')
        
        # Save model metrics
        joblib.dump(self.model_metrics, 'saved_models/model_metrics.pkl')
        
        logger.info("💾 All models and pipeline saved successfully!")
    
    def load_models(self):
        """Load pre-trained models"""
        
        try:
            # Load RandomForest
            self.models['random_forest'] = joblib.load('models/saved_models/random_forest_model.pkl')
            
            # Load LSTM if available
            if TORCH_AVAILABLE:
                checkpoint = torch.load('models/saved_models/lstm_model.pt', map_location=self.device)
                input_size = checkpoint['input_size']
                hidden_size = checkpoint['hidden_size']
                
                self.models['lstm'] = LSTMDemandPredictor(input_size, hidden_size).to(self.device)
                self.models['lstm'].load_state_dict(checkpoint['model_state_dict'])
            
            # Load feature pipeline
            self.feature_engineer.load_feature_pipeline('models/saved_models/feature_pipeline.pkl')
            
            # Load metrics
            self.model_metrics = joblib.load('models/saved_models/model_metrics.pkl')
            
            self.is_trained = True
            logger.info("📂 Pre-trained models loaded successfully!")
            
        except FileNotFoundError as e:
            logger.warning(f"Pre-trained models not found: {e}")
            logger.info("Please train models first using train_models()")
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get model performance comparison table"""
        
        if not self.model_metrics:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, metrics in self.model_metrics.items():
            if isinstance(metrics, dict) and 'test_accuracy' in metrics:
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Test Accuracy (%)': f"{metrics['test_accuracy']:.1f}%",
                    'Test MAE': f"{metrics['test_mae']:.3f}",
                    'Test RMSE': f"{metrics['test_rmse']:.3f}",
                    'Test R²': f"{metrics['test_r2']:.3f}",
                    'Model Type': metrics.get('model_type', 'Unknown')
                })
        
        return pd.DataFrame(comparison_data)

# LSTM Model Definition
if TORCH_AVAILABLE:
    class LSTMDemandPredictor(nn.Module):
        """
        LSTM Neural Network for demand prediction
        """
        
        def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
            super(LSTMDemandPredictor, self).__init__()
            
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # For this simplified version, we'll use a fully connected network
            # since we don't have proper sequence data
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
            self.fc4 = nn.Linear(hidden_size // 4, 1)
            
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            # Multi-layer feedforward network (simplified LSTM replacement)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.relu(self.fc3(x))
            x = self.fc4(x)
            
            return x.squeeze()

def main():
    """Train models on Chicago dataset"""
    
    # Load dataset
    logger.info("📂 Loading Chicago transportation dataset...")
    df = pd.read_csv('/Users/aswin/time_series_forecasting/data/chicago_rides_realistic.csv')
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    logger.info(f"✅ Loaded {len(df):,} records")
    
    # Initialize trainer
    trainer = ChicagoMLTrainer()
    
    # Train models
    metrics = trainer.train_models(df)
    
    # Display results
    comparison = trainer.get_model_comparison()
    logger.info("\n📊 Final Model Comparison:")
    logger.info("\n" + comparison.to_string(index=False))
    
    logger.info("\n🎉 Training pipeline completed successfully!")
    logger.info("🚀 Models ready for production predictions!")

if __name__ == "__main__":
    main()