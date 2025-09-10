"""
Real LSTM Model Architecture for Demand Prediction
This module contains the actual neural network architecture that will be trained on real data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class RealDemandLSTM(nn.Module):
    """
    Real LSTM model for ride demand prediction
    
    Architecture:
    - Multi-layer LSTM for temporal pattern learning
    - Feature embedding layers for categorical variables
    - Dropout for regularization
    - Output layer for demand prediction
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, embedding_dims: Dict[str, int] = None):
        super(RealDemandLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layers for categorical features
        self.embedding_dims = embedding_dims or {}
        self.embeddings = nn.ModuleDict()
        
        # Weather embedding (6 categories: clear, cloudy, light_rain, heavy_rain, snow, fog)
        if 'weather' in self.embedding_dims:
            self.embeddings['weather'] = nn.Embedding(6, self.embedding_dims['weather'])
        
        # Hour embedding (24 hours)
        if 'hour' in self.embedding_dims:
            self.embeddings['hour'] = nn.Embedding(24, self.embedding_dims['hour'])
        
        # Day of week embedding (7 days)
        if 'day_of_week' in self.embedding_dims:
            self.embeddings['day_of_week'] = nn.Embedding(7, self.embedding_dims['day_of_week'])
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            hidden: Optional hidden state tuple (h_0, c_0)
        
        Returns:
            Predicted demand values of shape (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            hidden = (h_0, c_0)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the last output from the sequence
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Apply fully connected layers
        output = self.fc_layers(last_output)
        
        # Apply ReLU to ensure non-negative demand
        output = torch.relu(output)
        
        return output.squeeze(-1)  # Shape: (batch_size,)


class GNNLSTMDemandModel(nn.Module):
    """
    Advanced Graph Neural Network + LSTM model for spatial-temporal demand prediction
    
    This model captures both spatial relationships between locations and temporal patterns.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 gnn_hidden_size: int = 64, num_locations: int = 100):
        super(GNNLSTMDemandModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gnn_hidden_size = gnn_hidden_size
        self.num_locations = num_locations
        
        # Location embedding
        self.location_embedding = nn.Embedding(num_locations, gnn_hidden_size)
        
        # Graph Neural Network layers (simplified as linear layers for now)
        self.gnn_layers = nn.Sequential(
            nn.Linear(gnn_hidden_size + 2, gnn_hidden_size),  # +2 for lat, lon
            nn.ReLU(),
            nn.Linear(gnn_hidden_size, gnn_hidden_size),
            nn.ReLU()
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_size + gnn_hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, location_ids: torch.Tensor, 
                lat_lon: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GNN-LSTM model
        
        Args:
            x: Input features (batch_size, sequence_length, input_size)
            location_ids: Location IDs (batch_size,)
            lat_lon: Latitude and longitude (batch_size, 2)
        
        Returns:
            Predicted demand (batch_size,)
        """
        batch_size, seq_len, _ = x.shape
        
        # Get location embeddings
        loc_embed = self.location_embedding(location_ids)  # (batch_size, gnn_hidden_size)
        
        # Apply GNN to incorporate spatial information
        spatial_input = torch.cat([loc_embed, lat_lon], dim=1)  # (batch_size, gnn_hidden_size + 2)
        spatial_features = self.gnn_layers(spatial_input)  # (batch_size, gnn_hidden_size)
        
        # Expand spatial features to match sequence length
        spatial_features = spatial_features.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, gnn_hidden_size)
        
        # Combine temporal and spatial features
        combined_input = torch.cat([x, spatial_features], dim=2)  # (batch_size, seq_len, input_size + gnn_hidden_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(combined_input)
        
        # Take last output
        last_output = lstm_out[:, -1, :]
        
        # Apply output layers
        output = self.output_layers(last_output)
        
        return torch.relu(output.squeeze(-1))


class MassiveScaleDemandLSTM(nn.Module):
    """
    WORKING SOLUTION: Colab-optimized LSTM for Chicago demand forecasting
    - Reduced model size for GPU memory constraints
    - Mixed precision training support
    - Efficient architecture for time series
    """
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.2):
        super(MassiveScaleDemandLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  # Reduced from 512 for Colab
        self.num_layers = num_layers    # Reduced from 3 for memory
        self.dropout = dropout
        
        # Input projection with batch norm
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM with reduced complexity
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Better pattern capture
        )
        
        # Attention mechanism (lightweight)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=4,  # Reduced from 8 for efficiency
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers (efficient design)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_size // 2, 1),
            nn.ReLU()  # Ensure non-negative demand
        )
        
        # Initialize weights efficiently
        self._init_weights()
        
        # Calculate model size
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'weight' in name and len(param.shape) >= 2:
                torch.nn.init.xavier_uniform_(param.data)
    
    def forward(self, x):
        """
        Forward pass optimized for both single and batch inputs
        Args:
            x: Input tensor (batch_size, input_size) or (batch_size, seq_len, input_size)
        Returns:
            Predicted demand (batch_size,)
        """
        batch_size = x.size(0)
        
        # Handle single time step inputs (non-sequential)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        seq_len = x.size(1)
        
        # Input projection
        if seq_len == 1:
            # Single time step
            x_proj = self.input_projection(x.squeeze(1))  # (batch_size, hidden_size)
            x_proj = x_proj.unsqueeze(1)  # (batch_size, 1, hidden_size)
        else:
            # Sequential input - apply projection to each time step
            x_reshaped = x.view(-1, x.size(-1))  # (batch_size * seq_len, input_size)
            x_proj = self.input_projection(x_reshaped)  # (batch_size * seq_len, hidden_size)
            x_proj = x_proj.view(batch_size, seq_len, -1)  # (batch_size, seq_len, hidden_size)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x_proj)
        # lstm_out: (batch_size, seq_len, hidden_size * 2) for bidirectional
        
        # Apply attention for sequence length > 1
        if seq_len > 1:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Combine with residual connection
            combined = lstm_out + attn_out
            # Pool over sequence dimension
            final_output = combined.mean(dim=1)  # (batch_size, hidden_size * 2)
        else:
            final_output = lstm_out.squeeze(1)  # (batch_size, hidden_size * 2)
        
        # Output layers
        output = self.output_layers(final_output)
        return output.squeeze(-1)  # (batch_size,)


# Backward compatibility alias
AdvancedDemandLSTM = MassiveScaleDemandLSTM


class FeatureExtractor:
    """
    Feature extraction utilities for real-world data preprocessing
    """
    
    @staticmethod
    def extract_temporal_features(timestamps: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract temporal features from timestamps"""
        import pandas as pd
        
        timestamps = pd.to_datetime(timestamps)
        
        features = {
            'hour': timestamps.hour.values,
            'day_of_week': timestamps.dayofweek.values,
            'month': timestamps.month.values,
            'day_of_year': timestamps.dayofyear.values,
            'is_weekend': (timestamps.dayofweek >= 5).astype(int).values,
            'is_rush_hour': ((timestamps.hour.isin([7, 8, 9, 17, 18, 19]))).astype(int).values,
            'is_business_hours': ((timestamps.hour >= 9) & (timestamps.hour <= 17)).astype(int).values,
            'is_night': ((timestamps.hour >= 22) | (timestamps.hour <= 5)).astype(int).values,
            
            # Cyclical encoding
            'hour_sin': np.sin(2 * np.pi * timestamps.hour / 24),
            'hour_cos': np.cos(2 * np.pi * timestamps.hour / 24),
            'day_sin': np.sin(2 * np.pi * timestamps.dayofweek / 7),
            'day_cos': np.cos(2 * np.pi * timestamps.dayofweek / 7),
            'month_sin': np.sin(2 * np.pi * timestamps.month / 12),
            'month_cos': np.cos(2 * np.pi * timestamps.month / 12),
        }
        
        return features
    
    @staticmethod
    def extract_spatial_features(latitudes: np.ndarray, longitudes: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract spatial features from coordinates"""
        
        # Distance from city center (using Chicago as example)
        city_center_lat, city_center_lon = 41.8781, -87.6298
        
        distance_from_center = np.sqrt(
            (latitudes - city_center_lat)**2 + (longitudes - city_center_lon)**2
        )
        
        # Distance from airport (O'Hare)
        airport_lat, airport_lon = 41.9742, -87.9073
        distance_from_airport = np.sqrt(
            (latitudes - airport_lat)**2 + (longitudes - airport_lon)**2
        )
        
        features = {
            'latitude': latitudes,
            'longitude': longitudes,
            'distance_from_center': distance_from_center,
            'distance_from_airport': distance_from_airport,
            'is_downtown': (distance_from_center < 0.05).astype(int),
            'is_airport_area': (distance_from_airport < 0.1).astype(int),
        }
        
        return features
    
    @staticmethod
    def encode_weather(weather_conditions: List[str]) -> np.ndarray:
        """One-hot encode weather conditions"""
        weather_mapping = {
            'clear': 0, 'cloudy': 1, 'light_rain': 2, 
            'heavy_rain': 3, 'snow': 4, 'fog': 5
        }
        
        encoded = np.zeros((len(weather_conditions), 6))
        
        for i, weather in enumerate(weather_conditions):
            if weather.lower() in weather_mapping:
                encoded[i, weather_mapping[weather.lower()]] = 1
            else:
                encoded[i, 0] = 1  # Default to clear
        
        return encoded
    
    @staticmethod
    def create_lag_features(demand_series: np.ndarray, lags: List[int] = [1, 24, 168]) -> Dict[str, np.ndarray]:
        """Create lag features for demand prediction"""
        import pandas as pd
        
        demand_df = pd.DataFrame({'demand': demand_series})
        lag_features = {}
        
        for lag in lags:
            lag_features[f'demand_lag_{lag}'] = demand_df['demand'].shift(lag).fillna(demand_df['demand'].mean()).values
        
        # Moving averages
        for window in [3, 12, 24]:
            lag_features[f'demand_ma_{window}'] = demand_df['demand'].rolling(window=window).mean().fillna(demand_df['demand'].mean()).values
        
        return lag_features

