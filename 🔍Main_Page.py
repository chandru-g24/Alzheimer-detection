import streamlit as st
import requests
from streamlit_lottie import st_lottie

st.set_page_config("Alzheimer's Detection",page_icon="🧠",layout='wide',initial_sidebar_state='collapsed')
video_html = """
<style>
[data-testid="stHeader"]{
   background-color: rgba(0,0,0,0);
}
"""

st.markdown(video_html, unsafe_allow_html=True)


@st.cache_resource(show_spinner="loading...")
def load_lottie():
    r = requests.get("https://assets7.lottiefiles.com/packages/lf20_wzq5hrhg.json")
    if r.status_code != 200:
        return None
    return r.json()
with st.sidebar:
    lottie_json = load_lottie()
    st_lottie = st_lottie(lottie_json,speed=2,loop=True,quality="high",height=300, width=300)

st.title("Detecting Alzheimer's through the lens of Convolutional Neural Networks 🔍")

st.info("Embark on an exciting journey of Alzheimer's detection with our cutting-edge deep learning model! 🚀🧠 By using advanced technology to analyze X-ray brain images, our groundbreaking solution accurately predicts the severity of the disease. It categorizes Alzheimer's into different levels: mild, moderate, very mild, and non-dementia. With an impressive accuracy rate of 96%, our model paves the way for early intervention and personalized care..")

st.subheader("Description about Alzheimer")

st.write("""Alzheimer's disease is a progressive neurological disorder that affects the brain, primarily causing memory loss and cognitive decline. It is the most common form of dementia, accounting for around **60-80%** of dementia cases. 

The disease is characterized by the presence of abnormal protein deposits, known as plaques and tangles, in the brain. These plaques are made up of beta-amyloid protein, while tangles consist of twisted strands of tau protein. Over time, these plaques and tangles disrupt the normal functioning of brain cells and lead to their gradual degeneration and death.

The symptoms of Alzheimer's disease typically start with mild memory loss and confusion, which gradually worsen over time. Other common symptoms include difficulty in language and communication, impaired judgment, disorientation, mood and personality changes, and challenges with problem-solving and performing daily tasks.

As Alzheimer's disease progresses, individuals may experience severe memory loss, lose the ability to recognize loved ones, and require round-the-clock care. The exact cause of Alzheimer's is not yet fully understood, but age, genetics, and certain lifestyle factors are believed to play a role in its development.

Currently, there is no cure for Alzheimer's disease. However, there are medications available that can temporarily alleviate symptoms and improve quality of life for some individuals. Non-pharmacological approaches, such as cognitive stimulation, physical exercise, and social engagement, may also have a positive impact on managing the symptoms and slowing down the progression of the disease.

Alzheimer's disease poses significant challenges for individuals affected by it and their families. It is crucial to raise awareness, promote early detection, and support ongoing research to develop effective treatments and, ultimately, find a cure for this devastating condition.""")