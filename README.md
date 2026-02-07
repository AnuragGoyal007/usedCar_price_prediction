# 🚗 Used Car Price Prediction

An end-to-end Machine Learning project that predicts the selling price of used cars based on key vehicle attributes.  
The project covers data cleaning, feature engineering, model selection, evaluation, and deployment as a web application.

---

## 📌 Project Overview

The objective of this project is to build a regression model that can accurately estimate the price of a used car.  
The dataset contains real-world car listings with both numerical and categorical attributes.

The final trained model is deployed as an interactive **Streamlit web application** where users can input car details and receive a predicted price.

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data Cleaning
- Converted price values from string format (₹, Lakh, Crore) into numeric values
- Cleaned numerical columns such as:
  - Kilometers driven
  - Engine capacity
  - Power (bhp)
  - Mileage
- Removed textual noise and handled missing values

---

### 2️⃣ Feature Engineering
Selected high-impact features based on domain knowledge and experimentation:

**Numerical Features**
- Manufacturing year (`myear`)
- Kilometers driven (`km`)
- Engine capacity (`engine_cc`)
- Power (`power_bhp`)
- Mileage (`mileage`)

**Categorical Feature**
- Transmission type (`tt`)

Categorical features were handled using **One-Hot Encoding** inside a preprocessing pipeline.

---

### 3️⃣ Model Selection
- **Random Forest Regressor** was chosen due to:
  - Ability to model non-linear relationships
  - Robustness to outliers
  - Strong performance on tabular data

Hyperparameters were tuned to balance bias and variance.

---

### 4️⃣ Model Evaluation

The model was evaluated using a hold-out test set.

**Performance Metrics:**
- **Mean Absolute Error (MAE):** ~1.23 lakh
- **R² Score:** ~0.75

This indicates the model explains approximately **75% of the variance** in car prices.

---

### 5️⃣ Model Interpretability
Feature importance analysis showed that:
- Engine capacity
- Power
- Mileage
- Manufacturing year

are the most influential predictors of car price, aligning well with real-world expectations.

---

## 🚀 Deployment

The trained model was saved using `joblib` and deployed as a **Streamlit web application**.

### 🔗 Live Demo
👉 *(Add your Streamlit app link here)*

Users can:
- Enter car details
- Instantly receive a predicted price (in Lakhs)

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 📂 Project Structure

car-price-prediction/
│
├── app.py # Streamlit application
├── car_price_model.pkl # Trained ML pipeline
├── requirements.txt # Project dependencies
├── README.md # Project documentation
├── cars_details_merged.csv # Dataset used
├── Car_price_prediction_CarDekho.ipynb # jupyter notebook


---

## ▶️ How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2. Install dependencies:
pip install -r requirements.txt

3. Run the app:
streamlit run app.py

---

📬 Contact
If you have feedback or suggestions, feel free to connect on LinkedIn or GitHub.