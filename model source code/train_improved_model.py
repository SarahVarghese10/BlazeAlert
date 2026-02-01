import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("5th iteration prediction sytstem w/ class weights and boosted weather features")
print("="*70)

# Training data file
TRAINING_DATA_FILE = 'merged_data_with_synthetic.csv'

# Model output files
MODEL_FILE = 'wildfire_model_improved.pth'
SCALER_FILE = 'wildfire_scaler_improved.pkl'
INFO_FILE = 'wildfire_model_info_improved.pkl'

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
PATIENCE = 15

# class weight - Missing a fire is 3x worse than a false alarm
FIRE_CLASS_WEIGHT = 3.0

print(f"\nTraining configuration:")
print(f"  Data file: {TRAINING_DATA_FILE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Max epochs: {NUM_EPOCHS}")
print(f"  Fire class weight: {FIRE_CLASS_WEIGHT}x")


# boost weather features to make model more responsive
MANUAL_WEIGHTS = {
    # weather (boosted)
    'Avg Air Temp (F)': 8.0,                                    
    'Avg Rel Hum (%)': 8.0,                                     
    'Avg Wind Speed (mph)': 7.0,                                
    'Month': 4.0,                                               
    
    # air quality (slightly boosted)
    'Daily AQI Value': 2.5,                                     
    'Daily Mean PM2.5 Concentration (ug/m3 LC)': 2.0,           
    
    # terrain/location (context only)
    'Landslide_Susceptibility': 0.8,                           
    'Fire_Prone_Area': 0.6,                                     
    
    # fire history (kept low to prevent location dominance)
    'Fire_Nearby_25mi': 0.5,                                    
    'Fires_Within_25mi': 0.5,                                   
    'Distance_To_Nearest_Fire_Miles': 0.5,                      
    'Total_Acres_Burned_25mi': 0.5,                            
    'Largest_Fire_Nearby_Acres': 0.2,                           
    'Fuel_Type': 0.5,                                           
    'Years_Since_Last_Fire': 0.3,                               
}

print("\n" + "="*70)
print("IMPROVED FEATURE WEIGHTS")
print("="*70)
print("\nWeather Features (BOOSTED):")
print(f"  Temperature:  {MANUAL_WEIGHTS['Avg Air Temp (F)']}x  (was 6.0x)")
print(f"  Humidity:     {MANUAL_WEIGHTS['Avg Rel Hum (%)']}x  (was 6.0x)")
print(f"  Wind Speed:   {MANUAL_WEIGHTS['Avg Wind Speed (mph)']}x  (was 5.0x)")
print(f"  Month:        {MANUAL_WEIGHTS['Month']}x  (was 3.5x)")
print("\nAir Quality (Slightly Boosted):")
print(f"  AQI:          {MANUAL_WEIGHTS['Daily AQI Value']}x  (was 2.0x)")
print(f"  PM2.5:        {MANUAL_WEIGHTS['Daily Mean PM2.5 Concentration (ug/m3 LC)']}x  (was 1.5x)")
print("\nLocation/History (Unchanged - Context Only):")
print(f"  Fire features: 0.2x - 0.6x  (kept low to prevent location dominance)")

# load training data
print("\n" + "="*70)
print("LOADING TRAINING DATA")
print("="*70)

print(f"\nReading: {TRAINING_DATA_FILE}")
try:
    df = pd.read_csv(TRAINING_DATA_FILE)
    print(f"Loaded: {len(df)} rows")
except FileNotFoundError:
    print(f"\nERROR: Could not find {TRAINING_DATA_FILE}")
    print("Please ensure the training data file is in the same directory.")
    exit(1)

# Check class distribution
fire_count = df['Fire_Reached'].sum()
no_fire_count = len(df) - fire_count

print(f"\nClass distribution:")
print(f"  Fire events: {fire_count} ({100*fire_count/len(df):.1f}%)")
print(f"  No-fire events: {no_fire_count} ({100*no_fire_count/len(df):.1f}%)")

#prep features

print("\n" + "="*70)
print("PREPARING FEATURES")
print("="*70)

# Get features in order defined by MANUAL_WEIGHTS
predictive_features = list(MANUAL_WEIGHTS.keys())
print(f"\nUsing {len(predictive_features)} features")

# Verify all features exist
missing_features = [f for f in predictive_features if f not in df.columns]
if missing_features:
    print(f"\nERROR: Missing features in data: {missing_features}")
    exit(1)

# Extract features and target
X = df[predictive_features].values
y = df['Fire_Reached'].values

print(f"Feature matrix: {X.shape}")
print(f"Target vector: {y.shape}")

#train/val split

print("\n" + "="*70)
print("TRAIN/VALIDATION SPLIT")
print("="*70)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"  Fire: {y_train.sum()} ({100*y_train.sum()/len(y_train):.1f}%)")
print(f"  No-fire: {len(y_train) - y_train.sum()} ({100*(len(y_train)-y_train.sum())/len(y_train):.1f}%)")

print(f"\nValidation set: {len(X_val)} samples")
print(f"  Fire: {y_val.sum()} ({100*y_val.sum()/len(y_val):.1f}%)")
print(f"  No-fire: {len(y_val) - y_val.sum()} ({100*(len(y_val)-y_val.sum())/len(y_val):.1f}%)")

#feature scaling

print("\n" + "="*70)
print("FEATURE SCALING")
print("="*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("StandardScaler fitted on training data")
print(f"Scaled training data: {X_train_scaled.shape}")
print(f"Scaled validation data: {X_val_scaled.shape}")

#nerual network with manual feature weighting

class ImprovedWildfireModel(nn.Module):
    
    def __init__(self, input_size, feature_weights):
        super(ImprovedWildfireModel, self).__init__()
        
        # Register feature weights
        self.register_buffer('feature_weights', torch.FloatTensor(feature_weights))
        
        # Network architecture
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1)
            # No sigmoid - outputs logits for BCEWithLogitsLoss
        )
    
    def forward(self, x):
        x_weighted = x * self.feature_weights
        return self.network(x_weighted)

# Initialize model
print("\n" + "="*70)
print("INITIALIZING MODEL")
print("="*70)

feature_weights = [MANUAL_WEIGHTS[f] for f in predictive_features]
model = ImprovedWildfireModel(len(predictive_features), feature_weights)

print(f"Model architecture:")
print(f"  Input: {len(predictive_features)} features (weighted)")
print(f"  Hidden: 64 -> 32 -> 16 neurons")
print(f"  Output: 1 neuron (logits)")

#training setup

print("\n" + "="*70)
print("TRAINING SETUP")
print("="*70)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_scaled)
y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# IMPORTANT: Use weighted loss function
pos_weight = torch.tensor([FIRE_CLASS_WEIGHT])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Loss function: BCEWithLogitsLoss")
print(f"  Positive class weight: {FIRE_CLASS_WEIGHT}x")
print(f"  (Missing a fire is {FIRE_CLASS_WEIGHT}x worse than false alarm)")
print(f"\nOptimizer: Adam")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Batch size: {BATCH_SIZE}")

#training loop

print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

best_f1 = 0
patience_counter = 0
best_model_state = None

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0
    
    for batch_features, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets in val_loader:
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            val_loss += loss.item()
            
            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.numpy())
            all_targets.extend(batch_targets.numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"  Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}")
    
    # Early stopping check
    if f1 > best_f1:
        best_f1 = f1
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
    
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        model.load_state_dict(best_model_state)
        break

print("\nTraining complete!")

#final evaluation

print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

model.eval()
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    val_probs = torch.sigmoid(val_outputs).numpy()
    val_preds = (val_probs > 0.5).astype(int)

# Calculate final metrics
accuracy = accuracy_score(y_val, val_preds)
precision = precision_score(y_val, val_preds, zero_division=0)
recall = recall_score(y_val, val_preds, zero_division=0)
f1_score_final = f1_score(y_val, val_preds, zero_division=0)
cm = confusion_matrix(y_val, val_preds)

print("\nPerformance Metrics:")
print(f"  Accuracy:  {accuracy:.2%}")
print(f"  Precision: {precision:.2%} (when predicting fire, how often correct)")
print(f"  Recall:    {recall:.2%} (% of actual fires caught)")
print(f"  F1-Score:  {f1_score_final:.2%}")

print("\nConfusion Matrix:")
print("                 Predicted")
print("              No Fire | Fire")
print(f"  Actual No:   {cm[0,0]:4d}   | {cm[0,1]:4d}")
print(f"  Actual Fire: {cm[1,0]:4d}   | {cm[1,1]:4d}")

# Probability analysis
fire_probs = val_probs[y_val == 1]
no_fire_probs = val_probs[y_val == 0]

print("\nProbability Distribution:")
print(f"  Fire cases - Avg: {fire_probs.mean():.1%}, Median: {np.median(fire_probs):.1%}")
print(f"  No-fire cases - Avg: {no_fire_probs.mean():.1%}, Median: {np.median(no_fire_probs):.1%}")
print(f"  Separation: {fire_probs.mean() - no_fire_probs.mean():.1%}")

#save model and info

print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

# Save model checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': epoch + 1,
    'best_f1': best_f1,
}, MODEL_FILE)
print(f"  Saved: {MODEL_FILE}")

# Save scaler
joblib.dump(scaler, SCALER_FILE)
print(f"  Saved: {SCALER_FILE}")

# Save model info
model_info = {
    'features': predictive_features,
    'feature_weights': MANUAL_WEIGHTS,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1_score_final,
    'fire_class_weight': FIRE_CLASS_WEIGHT,
}
joblib.dump(model_info, INFO_FILE)
print(f"  Saved: {INFO_FILE}")

#summary

print("\n" + "="*70)
print("training completeeeee")
print("="*70)

print(f"\nModel Performance:")
print(f"  accuracy:  {accuracy:.1%}")
print(f"  precision: {precision:.1%}")
print(f"  recall:    {recall:.1%}")  
print(f"  f1-Score:  {f1_score_final:.1%}")

print("="*70)