import streamlit as st
import numpy as np
from PIL import Image
import time
from scipy.spatial.distance import cdist


# Load data (cached for performance)
@st.cache_data
def read_data():
    all_vecs = np.load("all_vecs.npy")
    all_names = np.load("all_names.npy")
    return all_vecs, all_names

vecs, names = read_data()

# Custom CSS for UI improvements
st.markdown("""
    <style>
        /* Center everything */
        .block-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        
        /* Style buttons */
        .stButton > button {
            background-color: #007BFF !important; /* Blue */
            color: white !important;
            border-radius: 10px !important;
            padding: 10px 20px !important;
            font-size: 16px !important;
            border: none !important;
        }
        .stButton > button:hover {
            background-color: #0056b3 !important;
        }
        
        /* Image borders */
        img {
            border: 2px solid rgba(255, 255, 255, 0.5) !important;
            border-radius: 8px !important;
            box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.2) !important;
        }
    </style>
""", unsafe_allow_html=True)

# Layout
# Title and description (centered)
st.title("🔍 Image Similarity Finder")
st.write("Find similar ones from the database.") 


_ , fcol2 , _ = st.columns([1, 2, 1])  # Centering the image

scol1 , scol2 = st.columns([1, 1])  # Keeping buttons aligned

ch = scol1.button("🔄 Start / Change")
fs = scol2.button("🔍 Find Similar")

if ch:
    random_name = names[np.random.randint(len(names))]
    fcol2.image(Image.open("./images/" + random_name), caption="Selected Image", use_container_width=True)
    st.session_state["disp_img"] = random_name
    st.write(f"**Selected Image:** {st.session_state['disp_img']}")

if fs and "disp_img" in st.session_state:
    fcol2.image(Image.open("./images/" + st.session_state["disp_img"]), caption="Selected Image", use_container_width=True)
    
    idx = int(np.argwhere(names == st.session_state["disp_img"]))
    target_vec = vecs[idx]

    top5 = cdist(target_vec[None, ...], vecs).squeeze().argsort()[1:6]
    
    # Display similar images
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.image(Image.open("./images/" + names[top5[0]]), caption="Similar 1", use_container_width=True)
    c2.image(Image.open("./images/" + names[top5[1]]), caption="Similar 2", use_container_width=True)
    c3.image(Image.open("./images/" + names[top5[2]]), caption="Similar 3", use_container_width=True)
    c4.image(Image.open("./images/" + names[top5[3]]), caption="Similar 4", use_container_width=True)
    c5.image(Image.open("./images/" + names[top5[4]]), caption="Similar 5", use_container_width=True)
