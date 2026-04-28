import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Loads your saved model and feature names
@st.cache_resource
def load_model():
    pipe = joblib.load("model/churn_xgb_pipeline.pkl")
    feature_names = joblib.load("model/feature_names.pkl")
    return pipe, feature_names

pipe, feature_names = load_model()