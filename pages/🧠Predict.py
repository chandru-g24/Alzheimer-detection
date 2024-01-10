import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_lottie import st_lottie
import requests
from keras.models import model_from_json
import cv2
from PIL import Image, ImageOps

hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_resource(show_spinner="loading...")
def load_lottie():
    r = requests.get("https://assets10.lottiefiles.com/packages/lf20_w51pcehl.json")
    if r.status_code != 200:
        return None
    return r.json()
with st.sidebar:
    lottie_json = load_lottie()
    st_lottie = st_lottie(lottie_json,speed=1,loop=True,quality="high",height=400, width=300)

@st.cache_resource(show_spinner="Loading..", experimental_allow_widgets=True)
def model_():
        json_file = open('pages\model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        classifier = model_from_json(loaded_model_json)
        classifier.load_weights("pages\model_weights.h5")
        return classifier

model = model_()


col1, col2, col3 = st.columns([3 ,11, 3])

class_indices = {0: 'Mild Dementia', 1: 'Moderate Dementia',
                 2: 'Non Demented', 3: 'Very mild Dementia'}


with col2:
    st.title("Discover Your Outcome")
    file_ = st.file_uploader("Please upload the Brain X-ray image", type=['jpg','png','jpeg'])
    st.markdown("""___""")


if file_ is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = f"./temp_image.{file_.name.split('.')[-1]}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_.read())
    
    with col2:
        image = cv2.imread(temp_file_path)
        st.image(image,clamp=True,use_column_width=True)
        image = cv2.resize(image, (299,299))
        image = np.expand_dims(image, axis =0)
        image = image/255
        predictions = model.predict(image)
        result = np.argmax(predictions)
        probablity=np.max(predictions)
        st.success(
            f"Based on provide X-ray ,it seems to be {round(probablity*100,2)} % {class_indices[int(result)]}")

else:
    st.text(" ")