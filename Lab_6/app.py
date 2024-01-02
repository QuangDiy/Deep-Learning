import streamlit as st
from Exploration_Data import Exploration_Data
from VNFoods import VNFoods
from VSFC import VSFC

st.set_page_config(page_title="Lab 6 DeepLearning")

sidebar = st.sidebar
sidebar.title("Navigation")
page = sidebar.radio("Go to", ("Exploration Data", "VietNamese Foods", "Students Feedback"))

if page == "Exploration Data":
    Exploration_Data()
elif page == "Students Feedback":
    VSFC()
else:
    VNFoods()


