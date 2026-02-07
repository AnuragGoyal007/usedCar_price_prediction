import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import requests

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Used Car Price Prediction")
st.write("Predict the selling price of a used car (in Lakhs)")

# --------- Load & Train Model (Cached) ---------
DATA_URL = "https://raw.githubusercontent.com/AnuragGoyal007/usedCar_price_prediction/main/cars_details_merges.zip"


@st.cache_resource
def load_data():
    response = requests.get(DATA_URL)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open("cars_details_merges.csv") as f:
            return pd.read_csv(f)
        
@st.cache_resource        
def load_model():
    # Load dataset
    df = load_data()

    # ---- Cleaning ----
    def convert_price(x):
        x = str(x).replace('₹','').replace(',','').strip()
        if 'Crore' in x:
            return float(x.replace('Crore','')) * 100
        elif 'Lakh' in x:
            return float(x.replace('Lakh',''))
        else:
            return np.nan

    df['price'] = df['price'].apply(convert_price)

    df['km'] = (
        df['km']
        .str.replace('km','', regex=False)
        .str.replace(',','', regex=False)
        .str.strip()
    )
    df['km'] = pd.to_numeric(df['km'], errors='coerce')
    df['myear'] = pd.to_numeric(df['myear'], errors='coerce')

    def clean_numeric(col):
        return (
            col.astype(str)
            .str.replace('[^0-9.]','', regex=True)
            .replace('', np.nan)
            .astype(float)
        )

    df['engine_cc'] = clean_numeric(df['engine_cc'])

    df['power_bhp'] = (
        df['Max Power']
        .astype(str)
        .str.extract(r'([0-9]+\.?[0-9]*)')
        .astype(float)
    )

    df['mileage'] = (
        df['mileage_new']
        .astype(str)
        .str.extract(r'([0-9]+\.?[0-9]*)')
        .astype(float)
    )

    # ---- Feature Selection ----
    features = [
        'myear',
        'km',
        'engine_cc',
        'power_bhp',
        'mileage',
        'tt'
    ]

    df_model = df[features + ['price']].dropna()

    X = df_model.drop('price', axis=1)
    y = df_model['price']

    # ---- Pipeline ----
    categorical_cols = ['tt']
    numeric_cols = [
        'myear',
        'km',
        'engine_cc',
        'power_bhp',
        'mileage'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=25,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model)
    ])

    pipe.fit(X, y)
    return pipe

model = load_model()

# --------- User Inputs ---------
myear = st.number_input("Manufacturing Year", min_value=1995, max_value=2025, value=2018)
km = st.number_input("Kilometers Driven", min_value=0, value=40000)
engine_cc = st.number_input("Engine Capacity (cc)", min_value=500, value=1197)
power_bhp = st.number_input("Power (bhp)", min_value=30, value=82)
mileage = st.number_input("Mileage (kmpl)", min_value=5.0, value=18.0)
tt = st.selectbox("Transmission Type", ["Manual", "Automatic"])

# --------- Prediction ---------
if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        "myear": myear,
        "km": km,
        "engine_cc": engine_cc,
        "power_bhp": power_bhp,
        "mileage": mileage,
        "tt": tt
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Car Price: **₹ {prediction:.2f} Lakh**")
