import streamlit as st
import os, glob, pathlib, pickle, random, json, requests
from pathlib import Path
import pickle as pkle
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import tensorflow
import sklearn
import cv2
from PIL import Image
from tqdm import tqdm
from tensorflow.python.client import device_lib
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications
from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input
from st_clickable_images import clickable_images



os.environ['CUDA_VISIBLE_DEVICES']='2'


header = st.container()
with header:
    title = st.title('We recommend music that matches the image.')
    
    sh1 = st.container()
    with sh1:
        subheader1 = st.subheader('Informed Consent 📝')
        st.text("This is an Yonsei University research project.")
        st.text("All data for research is collected anonymously for research purposes.")
        st.text("For questions, please contact park.je@yonsei.ac.kr.")
        st.text("If you are under 18 years old, you need consent from your parents to participate.")
        st.text("Those who participate in the experiment are deemed to have consented to proceed with the experiment.")
        st.write('-----')

    sh2 = st.container()
    with sh2:
        subheader2 = st.subheader('What you are going to do 🧪')
        st.text("In this experiment, we recommend music that matches the image.") 
        st.text("We give you several kinds of image choices.")
        st.text("Select one image you like the most, then keywords will show up that match the selected image (up to 3).")
        st.text("If you are not satisfied with the keywords, you can change them from the selection.")
        st.text("When keyword selection is complete, we will automatically recommend musics that match the image.")
        st.write('-----')


    sh3 = st.container()
    with sh3:
        subheader3 = st.subheader('Warning 👀👂')
        st.text("If you SKIP A STEP, NO COMPENSATION will be provided.")
        st.write('-----')



    button = st.button('Agree, Start')
    # if button:
        
     
        


        









        
        
        


# header = st.container()
# with header:
#     title = st.title('We recommend music that matches the image.')
    
#     with st.form(key='consent', clear_on_submit=True):
#         subheader1 = st.subheader('Informed Consent 📝')
#         st.text("This is an Yonsei University research project.")
#         st.text("All data for research is collected anonymously for research purposes.")
#         st.text("For questions, please contact park.je@yonsei.ac.kr.")
#         st.text("If you are under 18 years old, you need consent from your parents to participate.")
#         st.text("Those who participate in the experiment are deemed to have consented to proceed with the experiment.")


#         subheader2 = st.subheader('What you are going to do 🧪')
#         st.text("In this experiment, we recommend music that matches the image.") 
#         st.text("We give you several kinds of image choices.")
#         st.text("Select one image you like the most, then keywords will show up that match the selected image (up to 3).")
#         st.text("If you are not satisfied with the keywords, you can change them from the selection.")
#         st.text("When keyword selection is complete, we will automatically recommend musics that match the image.")


#         subheader3 = st.subheader('Warning 👀👂')
#         st.text("If you SKIP A STEP, NO COMPENSATION will be provided.")
        
#         subheader4 = st.subheader('Agreement')
#         name = st.text_input("Name : ", help='Anonymous nickname is allowed.')
#         st.write('Do you agree to be involved in the experiment?')
#         agreement =  st.checkbox("Yes, I agree.")
#         print(agreement, "agreement value")

#         submitted = st.form_submit_button('SUBMIT')
#         print("submitted" if submitted=='True'else "Not submitted")
#         if submitted:
#             save_path = '/nas1/mingcha/img2music/retrieval/streamlit_v4/agreement.json'
#             results_B = {'Name': name, 'Agreement': agreement}
#             if not os.path.exists(save_path):
#                 data = {}
#                 data['submits'] = []
#                 data['submits'].append(results_B)
#                 print("no exists", data)
#                 with open(save_path, 'w') as save_f:
#                     json.dump(data, save_f, ensure_ascii=False, indent=4) 
#                 st.info('**The answer is successfully uploaded.**')           
        

