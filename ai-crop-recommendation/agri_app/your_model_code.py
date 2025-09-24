# --- Import necessary libraries ---
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import requests
import os

# --- Define base and data directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- Helper to normalize column names ---
def normalize_cols(df):
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    return df

# --- Load and normalize datasets ---
try:
    crop_df = normalize_cols(pd.read_csv(os.path.join(DATA_DIR, "crop_common_only_noman.csv")))
    market_df = normalize_cols(pd.read_csv(os.path.join(DATA_DIR, "market_common_only_noman.csv")))
    soil_df = normalize_cols(pd.read_excel(os.path.join(DATA_DIR, "india_cities_soil_full.xlsx")))
except Exception as e:
    print("❌ Error loading data files:", e)
    soil_df = pd.DataFrame()

# --- Set city column explicitly ---
soil_city_col = "city"

# --- Helper to guess column names based on keywords ---
def guess_col(df, keywords):
    for k in keywords:
        for col in df.columns:
            if k in col:
                return col
    return None

# --- Identify key columns dynamically ---
feature_cols = ['n','p','k','temperature','humidity','ph','rainfall']
crop_label_col = guess_col(crop_df, ['label','crop'])
commodity_col = guess_col(market_df, ['commodity','label','crop'])
modal_price_col = guess_col(market_df, ['modal_price','price','modal'])

# --- Prepare training data ---
X = crop_df[feature_cols]
y = crop_df[crop_label_col]

# --- Encode crop labels ---
le = LabelEncoder()
y_enc = le.fit_transform(y)

# --- Scale features for model training ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train RandomForest model with class balancing ---
rf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
rf.fit(X_scaled, y_enc)

# --- Lookup average market price for a crop ---
def get_market_price(crop_name):
    crop_name = crop_name.lower().strip()
    subset = market_df[market_df[commodity_col].str.lower() == crop_name]
    if not subset.empty and modal_price_col in subset.columns:
        return subset[modal_price_col].astype(float).mean()
    return np.nan

# --- Fetch live weather data from OpenWeather API ---
def get_weather(city_name):
    api_key = "92b96c68525f52eeb8a6215569b4d0e1"
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
        res = requests.get(url).json()

        # Debugging output
        print("Weather API response:", res)

        if "main" in res:
            temp = res['main']['temp']
            humidity = res['main']['humidity']
            return temp, humidity
        else:
            print("⚠️ Weather data missing for:", city_name)
            return None, None
    except Exception as e:
        print("❌ Weather API error:", e)
        return None, None

# --- Main crop recommendation function ---
def recommend_crop(city_input, yield_quintals=10):
    # Normalize city input and match against soil dataset
    city_input = city_input.strip().lower()
    soil_df[soil_city_col] = soil_df[soil_city_col].astype(str).str.strip().str.lower()
    soil_row = soil_df[soil_df[soil_city_col] == city_input]
    if soil_row.empty:
        return {"error": f"❌ Soil data for '{city_input}' not found."}

    # Extract soil features dynamically
    n_col = guess_col(soil_df, ['n','nitrogen'])
    p_col = guess_col(soil_df, ['p','phosphorus'])
    k_col = guess_col(soil_df, ['k','potassium'])
    ph_col = guess_col(soil_df, ['ph'])
    rainfall_col = guess_col(soil_df, ['rain','rainfall','rain_mm'])

    try:
        n = soil_row[n_col].values[0]
        p = soil_row[p_col].values[0]
        k = soil_row[k_col].values[0]
        ph = soil_row[ph_col].values[0]
        rainfall = soil_row[rainfall_col].values[0]
    except:
        return {"error": "⚠️ Missing soil features."}

    # Get live weather or fallback to dataset averages
    temp, humidity = get_weather(city_input)
    if temp is None:
        temp = crop_df['temperature'].mean()
        humidity = crop_df['humidity'].mean()

    # Prepare feature vector for prediction
    features = pd.DataFrame([[n, p, k, temp, humidity, ph, rainfall]], columns=feature_cols)
    features_scaled = scaler.transform(features)

    # Predict crop using trained model
    class_probs = rf.predict_proba(features_scaled)[0]
    best_idx = np.argmax(class_probs)
    best_crop = le.inverse_transform([best_idx])[0]

    # Estimate market price and profit
    price = get_market_price(best_crop)
    profit = yield_quintals * price if not np.isnan(price) else None

    # Try to get image filename from crop dataset
    image_col = guess_col(crop_df, ['image', 'image_file', 'image_name'])
    image_row = crop_df[crop_label_col].str.lower() == best_crop.lower()

    if image_col and not crop_df[image_row].empty:
        image_file = crop_df.loc[image_row, image_col].values[0]
    else:
        image_file = "default.jpg"

    # Return structured result
    return {
        "city": city_input.title(),
        "crop": best_crop,
        "price": round(price, 2) if price else "N/A",
        "profit": round(profit, 2) if profit else "N/A",
        "temp": round(temp, 1),
        "humidity": round(humidity, 1),
        "image": image_file,
        "soil": {
            "n": n, "p": p, "k": k, "ph": ph, "rainfall": rainfall
        }
    }

# --- Helper to get list of available cities for dropdown ---
def get_available_cities():
    if "city" not in soil_df.columns:
        return []
    cities = soil_df["city"].dropna().astype(str).str.strip()
    cities = cities[cities != ""]
    return sorted(cities.str.title().unique())

