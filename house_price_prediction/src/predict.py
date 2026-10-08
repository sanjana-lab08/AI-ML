import streamlit as st
import pandas as pd
import joblib
import shap

model = joblib.load("../models/house_price_model.pkl")
feature_columns = joblib.load("../models/feature_columns.pkl")

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Enter the details of the house to predict its price.")

area = st.number_input(
    "Area (sq ft)",
    value=5000
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    
    value=3,
    step=1
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    
    value=2,
    step=1
)

stories = st.number_input(
    "Number of Stories",
    
    value=2,
    step=1
)

mainroad = st.selectbox(
    "Main Road",
    ["yes", "no"]
)

guestroom = st.selectbox(
    "Guest Room",
    ["yes", "no"]
)

basement = st.selectbox(
    "Basement",
    ["yes", "no"]
)

hotwaterheating = st.selectbox(
    "Hot Water Heating",
    ["yes", "no"]
)

airconditioning = st.selectbox(
    "Air Conditioning",
    ["yes", "no"]
)

parking = st.number_input(
    "Parking Spaces",
    
    value=2,
    step=1
)

prefarea = st.selectbox(
    "Preferred Area",
    ["yes", "no"]
)

furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

input_data = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "stories": [stories],
    "mainroad": [mainroad],
    "guestroom": [guestroom],
    "basement": [basement],
    "hotwaterheating": [hotwaterheating],
    "airconditioning": [airconditioning],
    "parking": [parking],
    "prefarea": [prefarea],
    "furnishingstatus": [furnishingstatus]
})

input_data = pd.get_dummies(
    input_data,
    drop_first=True
)

input_data = input_data.reindex(
    columns=feature_columns,
    fill_value=0
)

if st.button("Predict House Price"):
    if area <= 0 or bedrooms < 0 or bathrooms < 0 or stories < 0 or parking < 0:
        st.error("Please enter valid values. Area must be greater than 0.")
    else:
        prediction = model.predict(input_data)

        st.success(
            f"Predicted House Price: ₹{prediction[0]:,.2f}"
        )

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_data)

        base_price = explainer.expected_value[0]
        contributions = shap_values.values[0]

        st.subheader("🏠 Price Explanation")

        st.write(f"**Base Model Price:** ₹{base_price:,.2f}")

        names = {
            "area": "Area",
            "bedrooms": "Bedrooms",
            "bathrooms": "Bathrooms",
            "stories": "Stories",
            "parking": "Parking",
            "mainroad_yes": "Main Road",
            "guestroom_yes": "Guest Room",
            "basement_yes": "Basement",
            "hotwaterheating_yes": "Hot Water Heating",
            "airconditioning_yes": "Air Conditioning",
            "prefarea_yes": "Preferred Area",
            "furnishingstatus_semi-furnished": "Furnishing Status",
            "furnishingstatus_unfurnished": "Furnishing Status"
        }

        report = []

        for feature, value in zip(feature_columns, contributions):
            if feature == "furnishingstatus_semi-furnished":
                continue

            if feature == "furnishingstatus_unfurnished":
                semi = contributions[
                    feature_columns.index("furnishingstatus_semi-furnished")
                ]
                value = value + semi

            report.append({
                "Factor": names[feature],
                "Contribution": f"₹{value:+,.2f}"
            })

        report = []

        for feature, value in zip(feature_columns, contributions):
            if feature == "furnishingstatus_semi-furnished":
                continue

            if feature == "furnishingstatus_unfurnished":
                semi = contributions[
                    feature_columns.index("furnishingstatus_semi-furnished")
                ]
                value = value + semi

            report.append({
                "Feature": names[feature],
                "Impact on Prediction": f"₹{abs(value):,.2f}"
            })

        report = pd.DataFrame(report)

        

        st.write(
            "The values below show how strongly each feature "
            "influenced the predicted price."
        )

        st.dataframe(
            report,
            hide_index=True,
            width="stretch"
        )