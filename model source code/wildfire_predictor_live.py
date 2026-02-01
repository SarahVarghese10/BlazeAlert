import torch
import torch.nn as nn
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.warp import transform as rasterio_transform
import joblib
import pgeocode
from datetime import datetime

#configuration

# API Keys for external data sources
API_KEYS = {
    'openweather': 'e113f6a46869fb8bf3a62a414f8ca62e',
    'airnow': 'D7524C14-4235-4BF0-A820-687E92DAF6C8',
    'nasa_firms': 'ee4e37728b63efd8fb41c42e7b2d3d5f'
}

# Risk level thresholds
RISK_LEVELS = {
    'LOW': (0, 20),
    'MODERATE': (20, 40),
    'ELEVATED': (40, 60),
    'HIGH': (60, 80),
    'EXTREME': (80, 100)
}

#live data fetchers

def get_current_weather(lat, lon):

    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': API_KEYS['openweather'],
        'units': 'imperial'  # Fahrenheit, mph
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return {
            'temp': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed']
        }
    except Exception as e:
        print(f"Warning: Could not fetch weather data: {e}")
        print("Using default weather values")
        # Return reasonable defaults if API fails
        return {
            'temp': 60,
            'humidity': 62,
            'wind_speed': 4.3
        }

def get_current_air_quality(zipcode):
   
    url = "http://www.airnowapi.org/aq/observation/zipCode/current/"
    params = {
        'format': 'application/json',
        'zipCode': zipcode,
        'distance': 25,  # Search within 25 miles
        'API_KEY': API_KEYS['airnow']
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # If no data from API, return defaults
        if not data:
            print(f"Warning: No air quality data available for zipcode {zipcode}")
            print("Using default air quality values")
            return {'pm25': 50, 'aqi': 50}
        
        # Extract PM2.5 and AQI values
        pm25 = None
        aqi = None
        
        for reading in data:
            if reading['ParameterName'] == 'PM2.5':
                pm25 = reading['AQI']
                aqi = reading['AQI']
                break
        
        # If PM2.5 not found, use first available AQI reading
        if pm25 is None and data:
            aqi = data[0]['AQI']
            pm25 = aqi * 0.5  # Rough conversion estimate
        
        return {
            'pm25': pm25 if pm25 else 50,
            'aqi': aqi if aqi else 50
        }
    except Exception as e:
        print(f"Warning: Could not fetch air quality: {e}")
        print("Using default air quality values")
        return {'pm25': 50, 'aqi': 50}

def get_active_fires_nearby(lat, lon, radius_miles=25):
   
    # NASA FIRMS API - area CSV endpoint
    # VIIRS_SNPP_NRT = VIIRS Suomi NPP Near Real-Time data
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{API_KEYS['nasa_firms']}/VIIRS_SNPP_NRT/{lat},{lon}/{radius_miles}/1"
    
    try:
        response = requests.get(url, timeout=10)
        
        # NASA FIRMS sometimes returns 500 errors (server issues)
        if response.status_code == 500:
            print("Warning: NASA FIRMS API temporarily unavailable")
            return 0
        
        response.raise_for_status()
        
        # Parse CSV response
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        if len(df) > 0:
            print(f"   ALERT: {len(df)} active fire(s) detected within {radius_miles} miles")
            return 1  # Active fire nearby
        else:
            return 0  # No active fires
            
    except Exception as e:
        print(f"Warning: Could not fetch active fire data: {str(e)[:100]}")
        return 0  # Assume no fires if API fails

#static data loader

class StaticDataLoader:
    
    
    def __init__(self):
        """Load all static data files on initialization."""
        print("Initializing static data loader...")
        
        # Load landslide susceptibility raster (terrain data)
        try:
            self.landslide_raster = rasterio.open('land_movement.tif')
            print("  Landslide susceptibility data loaded")
        except Exception as e:
            print(f"  Warning: Could not load landslide data: {e}")
            self.landslide_raster = None
        
        # Load fire perimeter GeoJSON (historical fires)
        try:
            self.fire_perimeters = gpd.read_file(
                'California_Historic_Fire_Perimeters_-5643920427189821347.geojson'
            )
            print(f"  Fire perimeter data loaded ({len(self.fire_perimeters)} fires)")
        except Exception as e:
            print(f"  Warning: Could not load perimeterrs: {e}")
            self.fire_perimeters = None
        
        # Load fuel type classification (vegetation)
        try:
            self.fuel_types = pd.read_csv('FUEL_VEGETATION_CSV_refined.csv')
            print("  Fuel type data loaded")
        except Exception as e:
            print(f"  Warning: Could not load fuel types: {e}")
            self.fuel_types = None
    
    def get_landslide_susceptibility(self, lat, lon):
    
        if self.landslide_raster is None:
            return 2.92  # Dataset mean as default
        
        try:
            # Convert lat/lon to raster coordinates
            src_crs = 'EPSG:4326'  # WGS84 (lat/lon)
            dst_crs = self.landslide_raster.crs.to_string()
            
            x, y = rasterio_transform(src_crs, dst_crs, [lon], [lat])
            
            # Get pixel row/column
            row, col = self.landslide_raster.index(x[0], y[0])
            
            # Read 10x10 window around point (handles edge cases)
            window_data = self.landslide_raster.read(
                1, 
                window=((row-5, row+5), (col-5, col+5))
            )
            
            # Return mean of window (smooths noise)
            return float(np.nanmean(window_data))
            
        except Exception:
            return 2.92  # Default to mean if extraction fails
    
    def get_fire_features(self, lat, lon):
       
        if self.fire_perimeters is None:
            # Return dataset averages as defaults
            return {
                'distance_to_nearest_fire': 1.78,
                'fires_within_25mi': 407,
                'total_acres_burned': 720427,
                'largest_fire_nearby': 50000,
                'years_since_last_fire': 5
            }
        
        try:
            # Convert fire perimeters to California Albers projection (meters)
            # This provides accurate distance measurements in meters
            fires_projected = self.fire_perimeters.to_crs('EPSG:3310')
            
            # Create point geometry and project to same CRS
            point = gpd.GeoSeries([Point(lon, lat)], crs='EPSG:4326')
            point_projected = point.to_crs('EPSG:3310').iloc[0]
            
            # Calculate distances in meters, convert to miles
            distances_meters = fires_projected.geometry.distance(point_projected)
            distances_miles = distances_meters / 1609.34  # meters to miles
            
            # Find fires within 25 miles
            nearby_fires = fires_projected[distances_miles <= 25]
            
            # Calculate metrics
            distance_to_nearest = distances_miles.min()
            fires_within_25mi = len(nearby_fires)
            
            # Calculate burned area
            if 'GIS_ACRES' in nearby_fires.columns:
                total_acres = nearby_fires['GIS_ACRES'].sum()
                largest_fire = nearby_fires['GIS_ACRES'].max() if len(nearby_fires) > 0 else 5000
            else:
                total_acres = fires_within_25mi * 1000  # Rough estimate
                largest_fire = 5000
            
            # Years since last fire (simplified - use fixed value)
            years_since = 5
            
            return {
                'distance_to_nearest_fire': distance_to_nearest,
                'fires_within_25mi': fires_within_25mi,
                'total_acres_burned': total_acres,
                'largest_fire_nearby': largest_fire,
                'years_since_last_fire': years_since
            }
            
        except Exception as e:
            print(f"Warning: Error calculating fire features: {e}")
            # Return defaults
            return {
                'distance_to_nearest_fire': 1.78,
                'fires_within_25mi': 407,
                'total_acres_burned': 720427,
                'largest_fire_nearby': 50000,
                'years_since_last_fire': 5
            }
    
    def get_fuel_type(self, zipcode):
       
        if self.fuel_types is None:
            return 2  # Default to shrub (most common in CA)
        
        try:
            # Look up zipcode in fuel type database
            match = self.fuel_types[self.fuel_types['Zipcode'] == int(zipcode)]
            
            if len(match) > 0:
                return int(match.iloc[0]['Fuel_Type'])
            else:
                return 2  # Default to shrub
                
        except Exception:
            return 2
    
    def is_fire_prone_area(self, lat, lon):
       
        return 1 if lat < 36 else 0


# neural network model with manual feature weighting

class ManualWeightedWildfireModel(nn.Module):
    
    def __init__(self, input_size, feature_weights):
        super(ManualWeightedWildfireModel, self).__init__()
        
        # Register feature weights as buffer (not trainable)
        self.register_buffer('feature_weights', torch.FloatTensor(feature_weights))
        
        # Define network architecture 15,64,32,16,1
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Apply manual weights before network processing
        x_weighted = x * self.feature_weights
        return self.network(x_weighted)

#wildfire predictor class
class WildfirePredictor:
    
    def __init__(self):
        print("\n" + "="*70)
        print("INITIALIZING WILDFIRE RISK PREDICTOR")
        print("="*70)
        
        # Load static data
        print("\nLoading static data files...")
        self.static_data = StaticDataLoader()
        
        # Load trained model
        print("\nLoading trained model...")
        self.load_model()
        
        # Initialize geocoder for zipcode lookup
        self.nomi = pgeocode.Nominatim('us')
        
        print("\n" + "="*70)
        print("PREDICTOR READY")
        print("="*70)
    
    def load_model(self):
        try:
            # Load model metadata (features, weights, metrics)
            model_info = joblib.load('wildfire_model_info_manual_weights.pkl')
            self.features = model_info['features']
            self.feature_weights_dict = model_info['feature_weights']
            
            # Extract weights in correct order
            feature_weights = [self.feature_weights_dict[f] for f in self.features]
            
            # Load feature scaler
            self.scaler = joblib.load('wildfire_scaler_manual_weights.pkl')
            
            # Initialize model architecture
            self.model = ManualWeightedWildfireModel(len(self.features), feature_weights)
            
            # Load trained weights
            checkpoint = torch.load(
                'wildfire_model_manual_weights.pth',
                weights_only=False
            )
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.model.eval()  # Set to evaluation mode
            
            print(f"  Model loaded successfully")
            print(f"    Features: {len(self.features)}")
            print(f"    Accuracy: {model_info['accuracy']*100:.1f}%")
            
        except Exception as e:
            print(f"\nERROR: Could not load model files")
            print(f"Details: {e}")
            print("\nPlease ensure these files exist:")
            print("  - wildfire_model_manual_weights.pth")
            print("  - wildfire_scaler_manual_weights.pkl")
            print("  - wildfire_model_info_manual_weights.pkl")
            exit(1)
    
    def predict(self, zipcode):
        
        print("\n" + "="*70)
        print(f"WILDFIRE RISK PREDICTION FOR ZIPCODE: {zipcode}")
        print("="*70)
        
        # Validate zipcode
        zip_info = self.nomi.query_postal_code(str(zipcode).zfill(5))
        
        if pd.isna(zip_info.latitude):
            print(f"\nERROR: Invalid zipcode: {zipcode}")
            print("Please enter a valid 5-digit California zipcode")
            return
        
        lat = zip_info.latitude
        lon = zip_info.longitude
        
        print(f"\nLocation: {zip_info.place_name}, {zip_info.state_code}")
        print(f"Coordinates: {lat:.4f}, {lon:.4f}")
        
        #fetch live data
        print("\nFetching live data from APIs...")
        
        weather = get_current_weather(lat, lon)
        air_quality = get_current_air_quality(zipcode)
        active_fire_nearby = get_active_fires_nearby(lat, lon)
        
        # load static data
        print("\nLoading location-specific data...")
        
        landslide = self.static_data.get_landslide_susceptibility(lat, lon)
        fire_features = self.static_data.get_fire_features(lat, lon)
        fuel_type = self.static_data.get_fuel_type(zipcode)
        fire_prone = self.static_data.is_fire_prone_area(lat, lon)
        
        # get current month
        current_month = datetime.now().month
        
        # combine all features into dict
        features_dict = {
            'Avg Air Temp (F)': weather['temp'],
            'Avg Rel Hum (%)': weather['humidity'],
            'Avg Wind Speed (mph)': weather['wind_speed'],
            'Daily Mean PM2.5 Concentration (ug/m3 LC)': air_quality['pm25'],
            'Daily AQI Value': air_quality['aqi'],
            'Month': current_month,
            'Fire_Nearby_25mi': active_fire_nearby,
            'Fire_Prone_Area': fire_prone,
            'Fuel_Type': fuel_type,
            'Landslide_Susceptibility': landslide,
            'Distance_To_Nearest_Fire_Miles': fire_features['distance_to_nearest_fire'],
            'Fires_Within_25mi': fire_features['fires_within_25mi'],
            'Total_Acres_Burned_25mi': fire_features['total_acres_burned'],
            'Largest_Fire_Nearby_Acres': fire_features['largest_fire_nearby'],
            'Years_Since_Last_Fire': fire_features['years_since_last_fire'],
        }
        
        # display current conditions
        print("\n" + "="*70)
        print("CURRENT CONDITIONS:")
        print("="*70)
        print(f"  Temperature:           {weather['temp']:.1f} F")
        print(f"  Humidity:              {weather['humidity']:.1f}%")
        print(f"  Wind Speed:            {weather['wind_speed']:.1f} mph")
        print(f"  Air Quality Index:     {air_quality['aqi']:.0f}")
        print(f"  PM2.5:                 {air_quality['pm25']:.1f}")
        print(f"  Month:                 {datetime.now().strftime('%B')}")
        print(f"  Active fires nearby:   {'YES' if active_fire_nearby else 'NO'}")
        print(f"  Terrain (0-10 scale):  {landslide:.1f}")
        print(f"  Historical fires:      {fire_features['fires_within_25mi']}")
        print(f"  Distance to nearest:   {fire_features['distance_to_nearest_fire']:.1f} mi")
        
        # predictions


        # Extract features in model's expected order
        feature_values = [features_dict[f] for f in self.features]
        feature_array = np.array([feature_values])
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_array)
        feature_tensor = torch.FloatTensor(feature_scaled)
        
        # Get prediction
        with torch.no_grad():
            probability = self.model(feature_tensor).item()
        
        # Convert to percentage
        risk_percentage = probability * 100
        
        # Determine risk level
        risk_level = None
        for level, (min_val, max_val) in RISK_LEVELS.items():
            if min_val <= risk_percentage < max_val:
                risk_level = level
                break
        if risk_percentage >= 80:
            risk_level = 'EXTREME'
        
       #results display
        print("\n" + "="*70)
        print("PREDICTION RESULT")
        print("="*70)
        print(f"\nFIRE RISK: {risk_percentage:.1f}%")
        print(f"RISK LEVEL: {risk_level}")
        print("\n" + "="*70)

# main execution

def main():
    """Main program loop."""
    
    # Display header
    print("="*70)
    print("WILDFIRE RISK PREDICTION SYSTEM")
    print("="*70)
    
    # Initialize predictor (loads model and data)
    predictor = WildfirePredictor()
    
    # Interactive prediction loop
    print("\n" + "="*70)
    print("predict wildfire risk by entering a California zipcode")
    print("="*70)
    print("\nEnter 'quit' to exit")
    
    while True:
        print("\n" + "-"*70)
        zipcode = input("\nEnter California zipcode (or 'quit'): ").strip()
        
        if zipcode.lower() in ['quit', 'exit', 'q']:
            print("\nexiting!!!")
            break
        
        if not zipcode.isdigit() or len(zipcode) != 5:
            print("ERROR: enter a valid zipcode")
            continue
        
        try:
            predictor.predict(zipcode)
        except Exception as e:
            print(f"\nERROR: Prediction failed")
            print(f"Details: {e}")
            print("try another zipode")

if __name__ == "__main__":
    main()