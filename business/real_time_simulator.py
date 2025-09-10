import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChicagoRealTimeSimulator:
    """
    Real-time simulation engine for Chicago transportation demand
    Simulates live updates, dynamic pricing, and fleet management
    """
    
    def __init__(self, ml_trainer=None):
        self.ml_trainer = ml_trainer
        self.is_running = False
        self.update_interval = 30  # seconds
        self.simulation_thread = None
        
        # Chicago locations with coordinates
        self.chicago_locations = {
            "Loop": (41.8781, -87.6298),
            "O'Hare Airport": (41.9742, -87.9073),
            "Magnificent Mile": (41.8955, -87.6244),
            "Lincoln Park": (41.9254, -87.6547),
            "Wicker Park": (41.9073, -87.6776),
            "Navy Pier": (41.8917, -87.6086),
            "Lakeview": (41.9403, -87.6438),
            "Logan Square": (41.9294, -87.7073),
            "Hyde Park": (41.7943, -87.5907),
            "Chinatown": (41.8508, -87.6320)
        }
        
        # Current state tracking
        self.current_demands = {}
        self.current_prices = {}
        self.driver_counts = {}
        self.wait_times = {}
        self.revenue_tracking = {}
        
        # Historical tracking for trends
        self.demand_history = {location: [] for location in self.chicago_locations}
        self.revenue_history = {location: [] for location in self.chicago_locations}
        
        # Initialize current state
        self._initialize_state()
        
        # Market events
        self.active_events = []
        
    def _initialize_state(self):
        """Initialize simulation state for all locations"""
        current_hour = datetime.now().hour
        
        for location, coords in self.chicago_locations.items():
            # Base demand based on location and time
            base_demand = self._calculate_base_demand(location, current_hour)
            
            self.current_demands[location] = max(1, int(base_demand + np.random.normal(0, 3)))
            self.current_prices[location] = self._calculate_base_price(self.current_demands[location])
            self.driver_counts[location] = max(1, self.current_demands[location] // 3 + np.random.randint(-2, 3))
            self.wait_times[location] = self._calculate_wait_time(
                self.current_demands[location], 
                self.driver_counts[location]
            )
            self.revenue_tracking[location] = 0.0
        
        logger.info("✅ Real-time simulation state initialized")
    
    def start_simulation(self):
        """Start the real-time simulation"""
        if self.is_running:
            logger.warning("Simulation already running")
            return
        
        self.is_running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
        logger.info("🚀 Real-time simulation started")
    
    def stop_simulation(self):
        """Stop the real-time simulation"""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join()
        logger.info("⏹️ Real-time simulation stopped")
    
    def _simulation_loop(self):
        """Main simulation loop running in background"""
        while self.is_running:
            try:
                self._update_market_conditions()
                self._simulate_demand_changes()
                self._update_pricing()
                self._simulate_driver_movements()
                self._track_revenue()
                self._check_events()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Simulation error: {e}")
                time.sleep(5)
    
    def _update_market_conditions(self):
        """Update market conditions based on time and external factors"""
        current_time = datetime.now()
        hour = current_time.hour
        day_of_week = current_time.weekday()
        
        # Rush hour multiplier
        rush_multiplier = 1.0
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            rush_multiplier = 1.5
        elif hour <= 5 or hour >= 22:
            rush_multiplier = 0.6
        
        # Weekend adjustment
        weekend_multiplier = 1.2 if day_of_week >= 5 else 1.0
        
        # Weather simulation (simplified)
        weather_conditions = ['clear', 'cloudy', 'light_rain', 'heavy_rain']
        weather_multipliers = [1.0, 1.1, 1.4, 1.8]
        current_weather = random.choice(list(zip(weather_conditions, weather_multipliers)))
        weather_multiplier = current_weather[1]
        
        # Apply conditions to each location
        for location in self.chicago_locations:
            location_base = self._calculate_base_demand(location, hour)
            adjusted_demand = location_base * rush_multiplier * weekend_multiplier * weather_multiplier
            
            # Add noise and ensure realistic bounds
            noise = np.random.normal(0, 2)
            self.current_demands[location] = max(1, min(80, int(adjusted_demand + noise)))
    
    def _simulate_demand_changes(self):
        """Simulate realistic demand fluctuations"""
        for location in self.chicago_locations:
            current_demand = self.current_demands[location]
            
            # Random walk with mean reversion
            change_magnitude = min(5, abs(np.random.normal(0, 2)))
            change_direction = np.random.choice([-1, 1])
            
            # Mean reversion (demands drift toward expected values)
            hour = datetime.now().hour
            expected_demand = self._calculate_base_demand(location, hour)
            if current_demand > expected_demand * 1.2:
                change_direction = -1  # Bias toward decrease
            elif current_demand < expected_demand * 0.8:
                change_direction = 1   # Bias toward increase
            
            new_demand = current_demand + (change_direction * change_magnitude)
            self.current_demands[location] = max(1, min(80, int(new_demand)))
            
            # Track history
            if len(self.demand_history[location]) > 100:
                self.demand_history[location].pop(0)
            self.demand_history[location].append(self.current_demands[location])
    
    def _update_pricing(self):
        """Update dynamic pricing based on demand"""
        for location in self.chicago_locations:
            demand = self.current_demands[location]
            
            # Surge pricing algorithm
            if demand >= 50:
                surge_multiplier = 2.2
            elif demand >= 40:
                surge_multiplier = 1.8
            elif demand >= 30:
                surge_multiplier = 1.4
            elif demand >= 20:
                surge_multiplier = 1.2
            else:
                surge_multiplier = 1.0
            
            base_price = 12.50  # Base Chicago fare
            self.current_prices[location] = round(base_price * surge_multiplier, 2)
    
    def _simulate_driver_movements(self):
        """Simulate driver allocation and movement"""
        total_demand = sum(self.current_demands.values())
        total_drivers = sum(self.driver_counts.values())
        
        # Redistribute drivers based on demand
        for location in self.chicago_locations:
            demand = self.current_demands[location]
            demand_ratio = demand / total_demand if total_demand > 0 else 0.1
            
            # Target driver count (with some inefficiency)
            target_drivers = max(1, int(total_drivers * demand_ratio * 0.8))
            current_drivers = self.driver_counts[location]
            
            # Gradual adjustment (drivers don't teleport)
            adjustment = int((target_drivers - current_drivers) * 0.3)
            self.driver_counts[location] = max(1, current_drivers + adjustment)
            
            # Update wait times
            self.wait_times[location] = self._calculate_wait_time(demand, self.driver_counts[location])
    
    def _track_revenue(self):
        """Track revenue generation"""
        for location in self.chicago_locations:
            demand = self.current_demands[location]
            price = self.current_prices[location]
            
            # Revenue per time interval
            interval_revenue = (demand * price * self.update_interval) / 3600  # Convert to hourly
            self.revenue_tracking[location] += interval_revenue
            
            # Track history
            if len(self.revenue_history[location]) > 100:
                self.revenue_history[location].pop(0)
            self.revenue_history[location].append(interval_revenue)
    
    def _check_events(self):
        """Check for special events that affect demand"""
        current_time = datetime.now()
        
        # Random event generation (5% chance every update)
        if random.random() < 0.05:
            event_types = [
                {"name": "Concert at United Center", "locations": ["Loop"], "multiplier": 1.8, "duration": 120},
                {"name": "Cubs Game", "locations": ["Lakeview"], "multiplier": 2.2, "duration": 180},
                {"name": "Convention at McCormick", "locations": ["Loop", "Magnificent Mile"], "multiplier": 1.5, "duration": 480},
                {"name": "Weather Alert", "locations": list(self.chicago_locations.keys()), "multiplier": 1.4, "duration": 90}
            ]
            
            event = random.choice(event_types)
            event["start_time"] = current_time
            event["end_time"] = current_time + timedelta(minutes=event["duration"])
            
            self.active_events.append(event)
            logger.info(f"🎪 New event: {event['name']}")
        
        # Apply active events
        for event in self.active_events[:]:
            if current_time <= event["end_time"]:
                # Event is active
                for location in event["locations"]:
                    if location in self.current_demands:
                        self.current_demands[location] = min(80, int(self.current_demands[location] * event["multiplier"]))
            else:
                # Event expired
                self.active_events.remove(event)
    
    def get_real_time_data(self) -> Dict:
        """Get current real-time market data"""
        current_time = datetime.now()
        
        # Calculate city-wide metrics
        total_demand = sum(self.current_demands.values())
        total_drivers = sum(self.driver_counts.values())
        avg_wait_time = np.mean(list(self.wait_times.values()))
        total_revenue = sum(self.revenue_tracking.values())
        
        # Active surge areas
        surge_areas = [
            location for location, price in self.current_prices.items() 
            if price > 15.0  # Base price * 1.2
        ]
        
        return {
            "timestamp": current_time.isoformat(),
            "city_metrics": {
                "total_demand": total_demand,
                "total_drivers": total_drivers,
                "utilization_rate": round((total_demand / total_drivers) * 100, 1) if total_drivers > 0 else 0,
                "avg_wait_time": round(avg_wait_time, 1),
                "total_revenue": round(total_revenue, 2),
                "active_surge_areas": len(surge_areas)
            },
            "location_data": {
                location: {
                    "demand": self.current_demands[location],
                    "price": self.current_prices[location],
                    "drivers": self.driver_counts[location],
                    "wait_time": self.wait_times[location],
                    "revenue": round(self.revenue_tracking[location], 2),
                    "coordinates": coords
                }
                for location, coords in self.chicago_locations.items()
            },
            "active_events": [
                {
                    "name": event["name"],
                    "locations": event["locations"],
                    "ends_in": str(event["end_time"] - current_time).split('.')[0]
                }
                for event in self.active_events
            ]
        }
    
    def get_demand_predictions(self) -> Dict:
        """Get ML predictions for next hour"""
        if not self.ml_trainer or not self.ml_trainer.is_trained:
            return {}
        
        predictions = {}
        next_hour = datetime.now() + timedelta(hours=1)
        
        for location, coords in self.chicago_locations.items():
            try:
                prediction = self.ml_trainer.predict(
                    lat=coords[0],
                    lon=coords[1],
                    timestamp=pd.Timestamp(next_hour),
                    model_type='random_forest'
                )
                
                predictions[location] = {
                    "predicted_demand": prediction['predicted_demand'],
                    "confidence": prediction['confidence'],
                    "expected_revenue": prediction['estimated_hourly_revenue']
                }
                
            except Exception as e:
                logger.error(f"Prediction failed for {location}: {e}")
                predictions[location] = {
                    "predicted_demand": self.current_demands[location],
                    "confidence": 0.5,
                    "expected_revenue": 0
                }
        
        return predictions
    
    def get_trend_data(self, location: str, hours: int = 2) -> Dict:
        """Get historical trend data for visualization"""
        if location not in self.demand_history:
            return {}
        
        demand_data = self.demand_history[location][-hours*2:] if len(self.demand_history[location]) >= hours*2 else self.demand_history[location]
        revenue_data = self.revenue_history[location][-hours*2:] if len(self.revenue_history[location]) >= hours*2 else self.revenue_history[location]
        
        return {
            "demand_trend": demand_data,
            "revenue_trend": revenue_data,
            "current_demand": self.current_demands[location],
            "current_price": self.current_prices[location]
        }
    
    def _calculate_base_demand(self, location: str, hour: int) -> float:
        """Calculate base demand for location and time"""
        # Location-specific base demands
        location_bases = {
            "Loop": 30, "O'Hare Airport": 35, "Magnificent Mile": 28,
            "Lincoln Park": 22, "Wicker Park": 20, "Navy Pier": 25,
            "Lakeview": 18, "Logan Square": 16, "Hyde Park": 14, "Chinatown": 12
        }
        
        base = location_bases.get(location, 15)
        
        # Time-based multipliers
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            return base * 1.6
        elif 10 <= hour <= 16:  # Business hours
            return base * 1.0
        elif 20 <= hour <= 23:  # Evening
            return base * 1.2
        else:  # Night/early morning
            return base * 0.4
    
    def _calculate_base_price(self, demand: int) -> float:
        """Calculate base price with surge"""
        base_price = 12.50
        
        if demand >= 40:
            return base_price * 1.8
        elif demand >= 30:
            return base_price * 1.4
        elif demand >= 20:
            return base_price * 1.2
        else:
            return base_price
    
    def _calculate_wait_time(self, demand: int, drivers: int) -> int:
        """Calculate estimated wait time"""
        if drivers == 0:
            return 15
        
        demand_per_driver = demand / drivers
        
        if demand_per_driver >= 5:
            return 12 + np.random.randint(0, 5)
        elif demand_per_driver >= 3:
            return 6 + np.random.randint(0, 4)
        else:
            return 2 + np.random.randint(0, 3)

# Example usage and testing
def main():
    """Test the real-time simulator"""
    simulator = ChicagoRealTimeSimulator()
    simulator.start_simulation()
    
    try:
        for i in range(10):
            data = simulator.get_real_time_data()
            print(f"\n⏰ Update {i+1}:")
            print(f"City Demand: {data['city_metrics']['total_demand']}")
            print(f"Total Revenue: ${data['city_metrics']['total_revenue']:.2f}")
            print(f"Active Events: {len(data['active_events'])}")
            
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("Stopping simulation...")
    
    finally:
        simulator.stop_simulation()

if __name__ == "__main__":
    main()