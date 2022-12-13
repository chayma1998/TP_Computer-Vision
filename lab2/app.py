import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict


st.title('Face detection')
st.markdown('TP CV')

st.header("Face Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Distance between the eyes")
    sepal_l = st.slider('lenght (mm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('width (mm)', 2.0, 4.4, 0.5)

with col2:
    st.text("Distance from the forehead to the chin")
    petal_l = st.slider('lenght (mm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('width (mm)', 0.1, 2.5, 0.5)

st.text('')
if st.button("Predict face"):
    result = predict(
        np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.text(result[0])


