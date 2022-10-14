from dbm import dumb
import streamlit as st
import joblib
import cv2
from PIL import Image
import numpy as np

# load model and load scaler
model = joblib.load('model.joblib')
model_1 = joblib.load('model_1.joblib')
model_2 = joblib.load('model_2.joblib')
model_3 = joblib.load('model_3.joblib')
#model_4 = joblib.load('model_4.joblib')
model_5 = joblib.load('model_5.joblib')
model_6 = joblib.load('model_6.joblib')

scaler_1 = joblib.load('scaler_1.joblib')
scaler_2 = joblib.load('scaler_2.joblib')
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

    

        image_scaler_1  = scaler_1.transform(img_resized)
        image_scaler_2  = scaler_2.transform(img_resized)

        image_pca = pca.transform(img_resized)

        image_sp1 = pca.transform(image_scaler_1)
        image_sp2 = pca.transform(image_scaler_2)

        st.text('predict su dung original model')
        st.write(int(model.predict(img_resized)))

        st.text('predict su dung min max scaler ')
        st.write(int(model_1.predict(image_scaler_1)))

        st.text('predict su dung standard scaler')
        st.write(int(model_2.predict(image_scaler_2)))

        st.text('predict su dung pca')
        st.write(int(model_3.predict(image_pca)))

        st.text('predict su dung min max scaler ket hop pca')
        st.write(int(model_5.predict(image_sp1)))

        st.text('predict su dung standard scaler ket hop pca')
        st.write(int(model_6.predict(image_sp2)))