from dbm import dumb
import streamlit as st
import joblib
import cv2
from PIL import Image
import numpy as np

# load model and load scaler
model = joblib.load('defaut.joblib')
model_modified = joblib.load('modified.joblib')
scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca.joblib')


st.title("Load image to predict")
folder = st.file_uploader("upload file", accept_multiple_files=True)

if folder is not None:
    for img in folder:
        # img = StringIO(img.getvalue().read()
    
        st.image(img, caption = 'Image', use_column_width=True)
        img=np.asarray(Image.open(img, mode = 'r'))

        st.write(img.shape[2])
        # convert img to gray if img is RGB
        if img.shape[2] != 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            


        # resize img

        img_resized = cv2.resize(img, [28,28], interpolation = cv2.INTER_AREA)
        st.image(img_resized, caption = 'Image resized to put into model', use_column_width=True)
        img_resized = img_resized.flatten().reshape(1,-1)

        st.text('predict su dung original model')
        st.write(int(model.predict(img_resized)))

        image_modified  = scaler.transform(img_resized)
        

        image_modified = pca.transform(image_modified)

        st.text('predict su dung standard scaler ket hop pca')
        st.write(int(model_modified.predict(image_modified)))