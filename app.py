import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource
def load_artifacts():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv('df_train.csv')

artifacts = load_artifacts()
df = load_data()

model    = artifacts['model']
scaler   = artifacts['scaler']
encoder  = artifacts['encoder']
num_cols = artifacts['num_cols']
cat_cols = artifacts['cat_cols']
feature_names = artifacts['feature_names']


st.title("Car Price Prediction")
st.write("EDA, prediction and model interpretation")

tab1, tab2, tab3 = st.tabs(["EDA", "Prediction", "Model Weights"])


with tab1:
    st.header("Exploratory Data Analysis")
    
    st.subheader("Dataset overview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head())
    
    st.subheader("Selling price distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df['selling_price'], bins=60, edgecolor='black')
    ax.set_xlabel("Selling price")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    
    st.subheader("Price vs numerical features")
    feature = st.selectbox("Choose feature", num_cols)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(df[feature], df['selling_price'], alpha=0.3)
    ax.set_xlabel(feature)
    ax.set_ylabel("Selling price")
    st.pyplot(fig)
    
    st.subheader("Average price by category")
    cat = st.selectbox("Choose categorical feature", cat_cols)
    avg_price = df.groupby(cat)['selling_price'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    avg_price.plot(kind='barh', ax=ax)
    ax.set_xlabel("Average price")
    st.pyplot(fig)
    
    st.subheader("Correlation matrix (numerical)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[num_cols + ['selling_price']].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)


with tab2:
    st.header("Prediction")
    
    mode = st.radio("Input mode", ["Manual input", "Upload CSV"])
    
    def preprocess(df_input):
        # num_cols_ordered = [
        # 'year', 'km_driven', 'mileage', 'engine', 
        # 'max_power', 'torque', 'seats', 'max_torque_rpm'
        # ]

        df_input = df_input.copy()
        if 'brand' not in df_input.columns:
            df_input['brand'] = df_input['name'].str.split().str[0]
            known = set(df['brand'].unique())
            df_input.loc[~df_input['brand'].isin(known), 'brand'] = 'Other'
        
        X_num = scaler.transform(df_input[num_cols])
        X_cat = encoder.transform(df_input[cat_cols])
        return np.hstack([X_num, X_cat])
    
    if mode == "Manual input":
        st.write("Enter car features:")
        
        col1, col2 = st.columns(2)
        user_input = {}
        
        with col1:
            user_input['year']           = st.number_input("Year", 1990, 2024, 2015)
            user_input['km_driven']      = st.number_input("Km driven", 0, 1000000, 50000)
            user_input['mileage']        = st.number_input("Mileage", 0.0, 50.0, 20.0)
            user_input['engine']         = st.number_input("Engine (cc)", 500, 5000, 1200)
            user_input['max_power']      = st.number_input("Max power", 0.0, 500.0, 80.0)
            user_input['torque']         = st.number_input("Torque", 0.0, 500.0, 100.0)
            user_input['max_torque_rpm'] = st.number_input("Max torque RPM", 1000.0, 8000.0, 2500.0)
        
        with col2:
            user_input['fuel']         = st.selectbox("Fuel",         df['fuel'].unique())
            user_input['seller_type']  = st.selectbox("Seller type",  df['seller_type'].unique())
            user_input['transmission'] = st.selectbox("Transmission", df['transmission'].unique())
            user_input['owner']        = st.selectbox("Owner",        df['owner'].unique())
            user_input['brand']        = st.selectbox("Brand",        df['brand'].unique())
            user_input['seats']        = st.selectbox("Seats",        sorted(df['seats'].unique()))
        
        if st.button("Predict"):
            df_input = pd.DataFrame([user_input])
            X = preprocess(df_input)
            st.write(X)
            pred = model.predict(X)[0]
            st.success(f"Predicted price: **{pred:,.0f}**")
    
    else:
        uploaded = st.file_uploader("Upload CSV", type='csv')
        if uploaded is not None:
            df_input = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(df_input.head())
            
            try:
                X = preprocess(df_input)
                preds = model.predict(X)
                df_input['predicted_price'] = preds
                
                st.success(f"Predicted {len(preds)} cars")
                st.dataframe(df_input)
                
                csv = df_input.to_csv(index=False).encode()
                st.download_button("Download predictions", csv, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error: {e}")


with tab3:
    st.header("Model coefficients")
    
    coefs = pd.DataFrame({
        'feature': feature_names,
        'coef':    model.coef_
    })
    coefs['abs_coef'] = coefs['coef'].abs()
    coefs = coefs.sort_values('abs_coef', ascending=False)
    
    st.subheader("Top features by absolute weight")
    top_n = st.slider("How many to show", 5, len(coefs), 15)
    
    top = coefs.head(top_n).sort_values('coef')
    
    fig, ax = plt.subplots(figsize=(10, top_n * 0.4))
    colors = ['red' if c < 0 else 'green' for c in top['coef']]
    ax.barh(top['feature'], top['coef'], color=colors)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Coefficient")
    st.pyplot(fig)
    
    st.subheader("All coefficients")
    st.dataframe(coefs.reset_index(drop=True))
    
    st.info(
        "🟢 Positive — increases price\n\n"
        "🔴 Negative — decreases price\n\n"
        "Larger absolute value → stronger effect on price."
    )