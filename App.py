import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("🏠 Real Estate Buyer Segmentation")
st.write("Machine Learning Based Buyer Segmentation and Investment Profiling")

# Sample real-estate buyer data
data = {
    "Buyer": ["A","B","C","D","E","F","G","H","I","J"],
    "Age": [25, 30, 45, 50, 28, 35, 55, 40, 32, 48],
    "Income_Lakh": [4, 5, 12, 15, 6, 8, 18, 10, 7, 14],
    "Budget_Lakh": [30, 40, 100, 150, 45, 70, 200, 90, 55, 130]
}

df = pd.DataFrame(data)

# K-Means clustering
X = df[["Income_Lakh", "Budget_Lakh"]]

model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Buyer_Segment"] = model.fit_predict(X)

# Investment profiles
profiles = {
    0: "Budget Buyer",
    1: "Mid-Range Investor",
    2: "Premium Investor"
}

df["Investment_Profile"] = df["Buyer_Segment"].map(profiles)

st.subheader("Buyer Segmentation Results")
st.dataframe(df)

# Visualization
st.subheader("Buyer Segmentation Visualization")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["Income_Lakh"], df["Budget_Lakh"])
ax.set_xlabel("Income (Lakh)")
ax.set_ylabel("Property Budget (Lakh)")
ax.set_title("Real Estate Buyer Segmentation")

st.pyplot(fig)
