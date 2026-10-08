# 🏠 House Price Prediction

A machine learning application that predicts house prices based on property characteristics.

## Project Overview

This project uses a machine learning model to predict house prices based on property features such as area, bedrooms, bathrooms, stories, parking, and other property characteristics.

The model is integrated with a Streamlit web application that allows users to enter property details and receive a predicted house price.

The application also provides a **price explanation** showing how strongly each property feature influenced the prediction.

## Features

* Area
* Number of bedrooms
* Number of bathrooms
* Number of stories
* Main road
* Guest room
* Basement
* Hot water heating
* Air conditioning
* Parking
* Preferred area
* Furnishing status
* Price prediction
* Feature-based price explanation

## Tech Stack

* Python
* Pandas
* Scikit-learn
* Joblib
* Streamlit
* SHAP

## Project Structure

```text
house_price_prediction/
│
├── models/
│   ├── house_price_model.pkl
│   └── feature_columns.pkl
│
├── src/
│   └── predict.py
│
├── requirements.txt
│
└── README.md
```

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Go to the `src` directory:

```bash
cd src
```

Run the Streamlit application:

```bash
streamlit run predict.py
```

The application will open in your browser.

## Price Explanation

After predicting the house price, the application displays the influence of each property feature on the prediction.

The displayed impact values indicate how strongly each feature influenced the model's prediction. They are **not additional costs or charges**.
