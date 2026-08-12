# ============================================================
# CAR PRICE PREDICTION
# Machine Learning + Gradio UI
# ============================================================

import os
import pandas as pd
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("car_data.csv")

print("Dataset:")
print(df.head())

print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns)


# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop("price_lakh", axis=1)

y = df["price_lakh"]


# ============================================================
# 3. DEFINE CATEGORICAL AND NUMERICAL COLUMNS
# ============================================================

categorical_columns = [
    "brand",
    "model",
    "fuel_type",
    "transmission"
]

numerical_columns = [
    "year",
    "km_driven",
    "engine_cc",
    "horsepower",
    "owner_count"
]


# ============================================================
# 4. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),

        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# ============================================================
# 5. CREATE MACHINE LEARNING MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# ============================================================
# 6. CREATE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

pipeline.fit(X_train, y_train)

print("\nModel training completed!")


# ============================================================
# 9. EVALUATE MODEL
# ============================================================

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)


print("\n========== MODEL PERFORMANCE ==========")

print("MAE :", mae)

print("MSE :", mse)

print("RMSE:", rmse)

print("R2  :", r2)


# ============================================================
# 10. PREDICTION FUNCTION FOR GRADIO
# ============================================================

def predict_price(
    brand,
    model_name,
    year,
    km_driven,
    fuel_type,
    transmission,
    engine_cc,
    horsepower,
    owner_count
):

    input_data = pd.DataFrame({
        "brand": [brand],
        "model": [model_name],
        "year": [year],
        "km_driven": [km_driven],
        "fuel_type": [fuel_type],
        "transmission": [transmission],
        "engine_cc": [engine_cc],
        "horsepower": [horsepower],
        "owner_count": [owner_count]
    })

    prediction = pipeline.predict(input_data)[0]

    return f"Estimated Car Price: ₹{prediction:.2f} Lakh"


# ============================================================
# 11. GRADIO UI
# ============================================================

interface = gr.Interface(

    fn=predict_price,

    inputs=[

        gr.Dropdown(
            choices=sorted(df["brand"].unique().tolist()),
            label="Brand"
        ),

        gr.Dropdown(
            choices=sorted(df["model"].unique().tolist()),
            label="Model"
        ),

        gr.Number(
            label="Manufacturing Year",
            value=2022
        ),

        gr.Number(
            label="Kilometers Driven",
            value=20000
        ),

        gr.Dropdown(
            choices=sorted(df["fuel_type"].unique().tolist()),
            label="Fuel Type"
        ),

        gr.Dropdown(
            choices=sorted(df["transmission"].unique().tolist()),
            label="Transmission"
        ),

        gr.Number(
            label="Engine CC",
            value=1500
        ),

        gr.Number(
            label="Horsepower",
            value=110
        ),

        gr.Number(
            label="Number of Previous Owners",
            value=1
        )
    ],

    outputs=gr.Textbox(
        label="Predicted Price"
    ),

    title="🚗 Car Price Prediction",

    description=(
        "Enter the car details below and the Machine Learning "
        "model will estimate its price."
    ),

    submit_btn="Predict Price",

    clear_btn="Clear"

)


# ============================================================
# 12. LAUNCH GRADIO APP
# ============================================================

port = int(os.environ.get("PORT", 8000))

interface.launch(
    server_name="0.0.0.0",
    server_port=port
)

