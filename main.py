import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

st.title('Outfit Recommender System')


# Ensure uploads folder exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load precomputed embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        print(e)
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocess_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocess_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File uploader
uploaded_file = st.file_uploader("Choose an image for outfit recommendation", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Image', use_container_width=True)

        # Feature extraction
        features = feature_extraction(os.path.join('uploads', uploaded_file.name), model)

        # Recommendation
        indices = recommend(features, feature_list)

        # Display recommendations
        st.subheader("Recommended Outfits:")
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            with col:
                st.image(filenames[indices[0][idx]], use_container_width=True)
    else:
        st.error("Error occurred during file upload.")
