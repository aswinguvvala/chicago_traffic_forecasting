import json
import os
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add src to path for model imports
sys.path.append('src')

class DynamicConfigLoader:
    """Dynamic configuration loader that calculates real values instead of hardcoding"""
    
    def __init__(self, config_path: str = "config/app_config.json"):
        self.config_path = config_path
        self.config = self._load_base_config()
        self.calculated_metrics = self._calculate_dynamic_metrics()
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback configuration if file doesn't exist
            return {
                "app": {"title": "Demand Forecasting", "demo_mode": True},
                "model": {"input_features": 8},
                "data": {"data_paths": []},
                "visualization": {"grid_resolution": 15}
            }
    
    def _calculate_dynamic_metrics(self) -> Dict[str, Any]:
        """Calculate actual metrics from data and model instead of hardcoding"""
        metrics = {}
        
        # Calculate real data statistics
        metrics['data_stats'] = self._calculate_data_statistics()
        
        # Get real model performance
        metrics['model_performance'] = self._get_model_performance()
        
        # Calculate system performance
        metrics['system_performance'] = self._calculate_system_performance()
        
        return metrics
    
    def _calculate_data_statistics(self) -> Dict[str, Any]:
        """Calculate actual data size and statistics"""
        stats = {
            "total_records": 0,
            "total_size_mb": 0,
            "files_processed": 0,
            "date_range": None
        }
        
        data_paths = self.config.get('data', {}).get('data_paths', [])
        
        for data_path in data_paths:
            if os.path.exists(data_path):
                try:
                    # Get file size
                    file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
                    stats['total_size_mb'] += file_size_mb
                    stats['files_processed'] += 1
                    
                    # Count records (read just first few rows to avoid loading everything)
                    df_sample = pd.read_csv(data_path, nrows=1000)
                    
                    # Estimate total records based on file size and sample
                    with open(data_path, 'r') as f:
                        line_count = sum(1 for _ in f) - 1  # Subtract header
                    
                    stats['total_records'] += line_count
                    
                except Exception as e:
                    print(f"Warning: Could not process {data_path}: {e}")
        
        # Format numbers appropriately 
        if stats['total_records'] >= 1000000:
            stats['records_display'] = f"{stats['total_records']/1000000:.1f}M"
        elif stats['total_records'] >= 1000:
            stats['records_display'] = f"{stats['total_records']/1000:.0f}K"
        else:
            stats['records_display'] = str(stats['total_records'])
            
        return stats
    
    def _get_model_performance(self) -> Dict[str, Any]:
        """Get actual model performance from trained model metadata"""
        performance = {
            "accuracy": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "r2_score": 0.0,
            "model_version": "Unknown",
            "status": "Not Loaded"
        }
        
        try:
            from models.simple_lstm_model import RealDemandPredictor
            
            model_path = self.config.get('model', {}).get('model_path', 'models/trained_demand_model.pt')
            predictor = RealDemandPredictor(model_path)
            
            if predictor.is_trained and hasattr(predictor, 'model_metadata'):
                metadata = predictor.model_metadata
                performance.update({
                    "accuracy": metadata.get('accuracy', 0.0),
                    "mae": metadata.get('mae', 0.0),
                    "rmse": metadata.get('rmse', 0.0),
                    "r2_score": metadata.get('r2_score', 0.0),
                    "model_version": metadata.get('version', 'LSTM-v1.0'),
                    "status": "Ready",
                    "training_samples": metadata.get('training_samples', 0),
                    "last_trained": metadata.get('last_trained', None)
                })
                
                # Calculate display accuracy (handle different accuracy formats)
                if performance["accuracy"] > 1:  # Already in percentage
                    performance["accuracy_display"] = f"{performance['accuracy']:.2f}%"
                elif performance["accuracy"] > 0:  # Decimal format
                    performance["accuracy_display"] = f"{performance['accuracy']*100:.2f}%"
                else:  # No accuracy available
                    performance["accuracy_display"] = "Training Required"
            else:
                performance["status"] = "Model Not Found"
                performance["accuracy_display"] = "No Model Loaded"
                
        except Exception as e:
            print(f"Warning: Could not load model performance: {e}")
            performance["status"] = "Error Loading Model"
            performance["accuracy_display"] = "Error"
            
        return performance
    
    def _calculate_system_performance(self) -> Dict[str, Any]:
        """Calculate system performance metrics"""
        return {
            "prediction_speed": "<2s",  # Could be measured dynamically
            "update_frequency": f"{self.config.get('app', {}).get('refresh_interval_seconds', 30)}s",
            "grid_resolution": self.config.get('visualization', {}).get('grid_resolution', 15)
        }
    
    def get_display_metrics(self) -> Dict[str, str]:
        """Get formatted metrics for UI display"""
        data_stats = self.calculated_metrics['data_stats']
        model_perf = self.calculated_metrics['model_performance']
        system_perf = self.calculated_metrics['system_performance']
        
        return {
            "accuracy": model_perf['accuracy_display'],
            "prediction_speed": system_perf['prediction_speed'],
            "data_size": data_stats['records_display'] + " Records",
            "revenue_increase": f"{self.config['business']['revenue_increase_target']}%",
            "model_status": model_perf['status'],
            "model_version": model_perf['model_version'],
            "total_files": str(data_stats['files_processed']),
            "update_frequency": system_perf['update_frequency']
        }
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get complete model metadata for detailed views"""
        return self.calculated_metrics['model_performance']
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get complete data statistics"""
        return self.calculated_metrics['data_stats']
    
    def is_demo_mode(self) -> bool:
        """Check if running in demo mode"""
        return self.config.get('app', {}).get('demo_mode', True)