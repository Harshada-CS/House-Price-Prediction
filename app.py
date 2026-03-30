import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

# Custom CSS (Premium UI 🔥)
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #141e30, #243b55);
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00f5c4;
}
.subtitle {
    text-align: center;
    color: #cccccc;
    margin-bottom: 20px;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 10px rgba(0,255,200,0.3);
}
.stNumberInput input {
    background-color: #2a2d34;
    color: white;
    border-radius: 10px;
}
.stButton>button {
    background: linear-gradient(90deg, #00f5c4, #00c6ff);
    color: black;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}
.result {
    text-align: center;
    background: linear-gradient(90deg, #00c6ff, #00f5c4);
    padding: 15px;
    border-radius: 15px;
    color: black;
    font-size: 22px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🏠 House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict house prices using Machine Learning</div>', unsafe_allow_html=True)

# Load data
df = pd.read_csv("data.csv")

X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

# Card layout
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("📐 Area (sqft)", min_value=500, value=1500)

with col2:
    bedrooms = st.number_input("🛏 Bedrooms", min_value=1, value=3)

bathrooms = st.number_input("🛁 Bathrooms", min_value=1, value=2)

if st.button("💰 Predict Price"):
    prediction = model.predict([[area, bedrooms, bathrooms]])

    st.markdown(f"""
        <div class="result">
            💵 Estimated Price: ₹ {int(prediction[0]):,}
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)