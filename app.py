import streamlit as st
import numpy as np
from prediction import predict

st.title('Face Detection')

st.text('')
if st.button("Process"):
    result = predict(
        np.array())
    st.text(result[0])

st.text('')
st.text('')

