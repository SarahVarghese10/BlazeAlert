from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import pgeocode

app = FastAPI(title="BlazeAlert Wildfire Risk API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API KEYS

API_KEYS = {
    'openweather': 'e113f6a46869fb8bf3a62a414f8ca62e',
    'airnow': 'D7524C14-4235-4BF0-A820-687E92DAF6C8',
    'nasa_firms': 'ee4e37728b63efd8fb41c42e7b2d3d5f'
}

# MODEL ARCHITECTURE

class ManualWeightedWildfireModel(nn.Module):
    """Neural network with manual feature weighting."""
    
    def __init__(self, input_size, feature_weights):
        super(ManualWeightedWildfireModel, self).__init__()
        
        self.register_buffer('feature_weights', torch.FloatTensor(feature_weights))
        
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
        x_weighted = x * self.feature_weights
        return self.network(x_weighted)

# LIVE DATA FETCHERS

def get_current_weather(lat, lon):
    """Fetch weather from OpenWeatherMap."""
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': API_KEYS['openweather'],
        'units': 'imperial'
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
    except:
        return {'temp': 70, 'humidity': 50, 'wind_speed': 5}

def get_current_air_quality(zipcode):
    """Fetch air quality from AirNow."""
    url = "http://www.airnowapi.org/aq/observation/zipCode/current/"
    params = {
        'format': 'application/json',
        'zipCode': zipcode,
        'distance': 25,
        'API_KEY': API_KEYS['airnow']
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return {'pm25': 50, 'aqi': 50}
        
        pm25 = None
        aqi = None
        
        for reading in data:
            if reading['ParameterName'] == 'PM2.5':
                pm25 = reading['AQI']
                aqi = reading['AQI']
                break
        
        if pm25 is None and data:
            aqi = data[0]['AQI']
            pm25 = aqi * 0.5
        
        return {
            'pm25': pm25 if pm25 else 50,
            'aqi': aqi if aqi else 50
        }
    except:
        return {'pm25': 50, 'aqi': 50}

def get_active_fires_nearby(lat, lon, radius_miles=25):
    """Check for active fires using NASA FIRMS."""
    radius_deg = radius_miles / 69.0
    north = lat + radius_deg
    south = lat - radius_deg
    east = lon + radius_deg
    west = lon - radius_deg
    
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{API_KEYS['nasa_firms']}/VIIRS_SNPP_NRT/{west},{south},{east},{north}/1"
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return {'active': False, 'count': 0, 'details': []}
        
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        if len(df) == 0:
            return {'active': False, 'count': 0, 'details': []}
        
        # Find brightness column
        brightness_col = None
        for col in ['bright_ti4', 'brightness', 'bright_ti5', 'Bright_ti4', 'bright_t31']:
            if col in df.columns:
                brightness_col = col
                break
        
        # Return all detections (removed strict filtering)
        fire_details = []
        for idx, fire in df.head(10).iterrows():
            fire_details.append({
                'latitude': float(fire['latitude']),
                'longitude': float(fire['longitude']),
                'brightness': float(fire[brightness_col]) if brightness_col else 0
            })
        
        return {
            'active': len(df) > 0,
            'count': len(df),
            'details': fire_details
        }
            
    except Exception as e:
        print(f"FIRMS error: {e}")
        return {'active': False, 'count': 0, 'details': []}

def get_all_california_fires():
    """Get ALL active fires in California for map display."""
    # California bounding box
    north, south = 42.0, 32.5
    west, east = -124.5, -114.0
    
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{API_KEYS['nasa_firms']}/VIIRS_SNPP_NRT/{west},{south},{east},{north}/1"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return []
        
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        if len(df) == 0:
            return []
        
        fires = []
        for idx, fire in df.iterrows():
            fires.append({
                'lat': float(fire['latitude']),
                'lon': float(fire['longitude']),
                'brightness': float(fire.get('bright_ti4', fire.get('brightness', 0)))
            })
        
        return fires
    except:
        return []

# PYDANTIC MODELS

class PredictRequest(BaseModel):
    zipcode: str

class PredictionResponse(BaseModel):
    risk_score: float
    risk_level: str
    latitude: float
    longitude: float
    active_fires_detected: bool
    fire_count: int
    contributing_factors: dict
    timestamp: str

# GLOBAL MODEL LOADER

model = None
scaler = None
features = None
nomi = pgeocode.Nominatim('us')

try:
    import joblib
    
    model_info = joblib.load('wildfire_model_info_manual_weights.pkl')
    features = model_info['features']
    feature_weights = [model_info['feature_weights'][f] for f in features]
    
    scaler = joblib.load('wildfire_scaler_manual_weights.pkl')
    
    model = ManualWeightedWildfireModel(len(features), feature_weights)
    checkpoint = torch.load('wildfire_model_manual_weights.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(" Model loaded successfully!")
    print(f"   Features: {len(features)}")
    print(f"   Accuracy: {model_info['accuracy']*100:.1f}%")
    
    # DEBG: Test model
    print("\n🧪 Testing model...")
    test_input = torch.FloatTensor([[0.5] * 15])
    test_output = model(test_input)
    print(f"   Test prediction: {test_output.item() * 100:.2f}%")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")

# API ENDPOINTS

@app.get("/")
async def root():
    return {
        "service": "BlazeAlert Wildfire Risk API",
        "status": "online",
        "model_loaded": model is not None,
        "version": "3.0 (Full Model - No Bandaid)"
    }

@app.get("/california-fires")
async def get_california_fires():
    """Get all active fires in California for map display."""
    fires = get_all_california_fires()
    return {
        "fires": fires,
        "count": len(fires),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_wildfire_risk(request: PredictRequest):
    """Predict wildfire risk using the full neural network model."""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Geocode zipcode
        zip_info = nomi.query_postal_code(str(request.zipcode).zfill(5))
        
        if pd.isna(zip_info.latitude):
            raise HTTPException(status_code=400, detail="Invalid zipcode")
        
        lat = zip_info.latitude
        lon = zip_info.longitude
        
        # Fetch live data
        weather = get_current_weather(lat, lon)
        air_quality = get_current_air_quality(request.zipcode)
        fire_detection = get_active_fires_nearby(lat, lon)
        
        current_month = datetime.now().month
        
        # Prepare ALL 15 features
        features_dict = {
            'Avg Air Temp (F)': weather['temp'],
            'Avg Rel Hum (%)': weather['humidity'],
            'Avg Wind Speed (mph)': weather['wind_speed'],
            'Daily Mean PM2.5 Concentration (ug/m3 LC)': air_quality['pm25'],
            'Daily AQI Value': air_quality['aqi'],
            'Month': current_month,
            'Fire_Nearby_25mi': 1 if fire_detection['active'] else 0,
            'Fire_Prone_Area': 1 if lat < 36 else 0,  # Southern CA
            'Fuel_Type': 2,  # Default shrub
            'Landslide_Susceptibility': 2.92,  # Default terrain
            'Distance_To_Nearest_Fire_Miles': 5.0 if fire_detection['active'] else 20.0,
            'Fires_Within_25mi': fire_detection['count'],
            'Total_Acres_Burned_25mi': fire_detection['count'] * 500,  # Estimate
            'Largest_Fire_Nearby_Acres': 500 if fire_detection['active'] else 0,
            'Years_Since_Last_Fire': 2,  # Default
        }
        
        # Prepare model input
        feature_values = [features_dict[f] for f in features]
        feature_array = np.array([feature_values])
        feature_scaled = scaler.transform(feature_array)
        feature_tensor = torch.FloatTensor(feature_scaled)
        
        # Get prediction from neural network
        with torch.no_grad():
            probability = model(feature_tensor).item()
        
        risk_score = probability * 100
        
        # Determine risk level
        if risk_score >= 80:
            risk_level = 'EXTREME'
        elif risk_score >= 60:
            risk_level = 'HIGH'
        elif risk_score >= 40:
            risk_level = 'ELEVATED'
        elif risk_score >= 20:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
        
        # Build ALL 15 contributing factors for display
        contributing_factors = {
            'temperature': {
                'value': weather['temp'],
                'impact': 'high' if weather['temp'] > 85 else 'medium' if weather['temp'] > 75 else 'low',
                'message': f'Temperature: {weather["temp"]:.1f}°F - {"Elevated" if weather["temp"] > 85 else "Normal"} conditions'
            },
            'humidity': {
                'value': weather['humidity'],
                'impact': 'high' if weather['humidity'] < 30 else 'medium' if weather['humidity'] < 50 else 'low',
                'message': f'Humidity: {weather["humidity"]:.1f}% - {"Very dry" if weather["humidity"] < 30 else "Moderate"} moisture'
            },
            'wind_speed': {
                'value': weather['wind_speed'],
                'impact': 'high' if weather['wind_speed'] > 15 else 'medium' if weather['wind_speed'] > 10 else 'low',
                'message': f'Wind: {weather["wind_speed"]:.1f} mph - {"Strong" if weather["wind_speed"] > 15 else "Moderate"} winds'
            },
            'pm25': {
                'value': air_quality['pm25'],
                'impact': 'high' if air_quality['pm25'] > 100 else 'medium' if air_quality['pm25'] > 50 else 'low',
                'message': f'PM2.5: {air_quality["pm25"]:.0f} μg/m³ - {"Poor" if air_quality["pm25"] > 100 else "Moderate"} air quality'
            },
            'air_quality': {
                'value': air_quality['aqi'],
                'impact': 'high' if air_quality['aqi'] > 150 else 'medium' if air_quality['aqi'] > 100 else 'low',
                'message': f'AQI: {air_quality["aqi"]:.0f} - {"Unhealthy" if air_quality["aqi"] > 150 else "Moderate"}'
            },
            'active_fires': {
                'value': fire_detection['count'],
                'impact': 'critical' if fire_detection['count'] > 0 else 'low',
                'message': f'{fire_detection["count"]} active fire(s) detected within 25 miles' if fire_detection['count'] > 0 else 'No active fires detected'
            },
            'fire_nearby': {
                'value': features_dict['Fire_Nearby_25mi'],
                'impact': 'high' if features_dict['Fire_Nearby_25mi'] == 1 else 'low',
                'message': 'Fire detected in 25-mile radius' if features_dict['Fire_Nearby_25mi'] == 1 else 'No fires in immediate area'
            },
            'fire_prone_area': {
                'value': features_dict['Fire_Prone_Area'],
                'impact': 'medium' if features_dict['Fire_Prone_Area'] == 1 else 'low',
                'message': 'High fire-prone region' if features_dict['Fire_Prone_Area'] == 1 else 'Moderate fire history'
            },
            'fuel_type': {
                'value': features_dict['Fuel_Type'],
                'impact': 'medium',
                'message': f'Vegetation type: {features_dict["Fuel_Type"]} (shrub/grassland)'
            },
            'terrain': {
                'value': features_dict['Landslide_Susceptibility'],
                'impact': 'medium' if features_dict['Landslide_Susceptibility'] > 3 else 'low',
                'message': f'Terrain steepness: {features_dict["Landslide_Susceptibility"]:.1f} - {"Steep" if features_dict["Landslide_Susceptibility"] > 3 else "Moderate"} slope'
            },
            'distance_to_fire': {
                'value': features_dict['Distance_To_Nearest_Fire_Miles'],
                'impact': 'high' if features_dict['Distance_To_Nearest_Fire_Miles'] < 10 else 'medium' if features_dict['Distance_To_Nearest_Fire_Miles'] < 20 else 'low',
                'message': f'{features_dict["Distance_To_Nearest_Fire_Miles"]:.1f} miles to nearest fire'
            },
            'fires_within_25mi': {
                'value': features_dict['Fires_Within_25mi'],
                'impact': 'high' if features_dict['Fires_Within_25mi'] > 2 else 'medium' if features_dict['Fires_Within_25mi'] > 0 else 'low',
                'message': f'{features_dict["Fires_Within_25mi"]} fire(s) within 25-mile radius'
            },
            'acres_burned': {
                'value': features_dict['Total_Acres_Burned_25mi'],
                'impact': 'high' if features_dict['Total_Acres_Burned_25mi'] > 10000 else 'medium' if features_dict['Total_Acres_Burned_25mi'] > 1000 else 'low',
                'message': f'{features_dict["Total_Acres_Burned_25mi"]:,.0f} acres burned historically'
            },
            'largest_fire': {
                'value': features_dict['Largest_Fire_Nearby_Acres'],
                'impact': 'high' if features_dict['Largest_Fire_Nearby_Acres'] > 5000 else 'medium' if features_dict['Largest_Fire_Nearby_Acres'] > 500 else 'low',
                'message': f'Largest nearby fire: {features_dict["Largest_Fire_Nearby_Acres"]:,.0f} acres'
            },
            'years_since_fire': {
                'value': features_dict['Years_Since_Last_Fire'],
                'impact': 'high' if features_dict['Years_Since_Last_Fire'] > 10 else 'medium' if features_dict['Years_Since_Last_Fire'] > 5 else 'low',
                'message': f'{features_dict["Years_Since_Last_Fire"]} years since last fire - {"High" if features_dict["Years_Since_Last_Fire"] > 10 else "Moderate"} fuel accumulation'
            },
            'season': {
                'value': 'Summer' if 6 <= current_month <= 9 else 'Winter',
                'impact': 'high' if 6 <= current_month <= 9 else 'low',
                'message': 'Peak wildfire season (Summer/Fall)' if 6 <= current_month <= 9 else 'Lower risk season (Winter/Spring)'
            }
        }
        
        return PredictionResponse(
            risk_score=round(risk_score, 1),
            risk_level=risk_level,
            latitude=lat,
            longitude=lon,
            active_fires_detected=fire_detection['active'],
            fire_count=fire_detection['count'],
            contributing_factors=contributing_factors,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)