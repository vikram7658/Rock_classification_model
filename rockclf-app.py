# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 09:22:42 2020

@author: Vikram
"""
import pandas as pd
import numpy as np
import streamlit as st
import pickle
from PIL import Image 


image = Image.open("nepal.jpg")  
st.image(image, use_column_width=True)

pickle_in = open("Rockclass.pkl", "rb")
classifier = pickle.load(pickle_in)

df = pd.read_csv("rockclass.csv", usecols=['D', 'H', 'Q', 'k', 'Class'])
num_columns = ['D','H','Q','k']
X = df.drop(columns=['Class'], axis=1)



uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


def rock_class(D, Q, k, H):
    """lets classify the rock based on 
    this 4 parameter
    
    """                    
    Prediction = classifier.predict([[D, Q, k, H]])
    
    print(Prediction)
    return Prediction

def rock_data():
    """lets classify the rock based on 
    this 4 parameter
    
    """
    data_grp = pd.read_csv(uploaded_file, usecols=['D', 'H', 'Q', 'k'])                    
    Prediction = classifier.predict(data_grp)
    
    print(Prediction)
    return Prediction

def main():
    #decoration of the app
    st.title(" Rock Class Prediction")
    
    st.write("Prepared by **Vikram**. For code behind this ML app visit: https://github.com/vikram7658/Rock-Analysis")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Rock Class Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    #Data Input section creation
    D = st.text_input("D (m)", "Type diameter of tunnel here")
    Q = st.text_input("Q ", "Type Q value here")
    k = st.text_input("k (MPa)", "Type Support stiffness here")
    H = st.text_input("H (m)", "Type depth of overburden here")

    result = ""
    if st.button("predict_value"):
        result = rock_class(D, Q, k, H)
    elif st.button("predict_data"):
        result = rock_data()
        
    if st.success("The output is {}".format(result)) == 0:
        value= print("rock is minor type")
        st.write(value)
    st.subheader("Squeezing grade = *0: minor , 1:mild, 2:sever*")
    if st.button("About"):
        st.text("Data Retrieved from ://doi.org/10.1155/2018/4543984")
        st.text("Model code access :https://github.com/vikram7658/Rock-Analysis")
    
    st.subheader("Example of data type used in model")
    st.write(X)  
      
        
if __name__=='__main__':
    main()