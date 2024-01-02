import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st


model = load_model('model/base_model_trained.h5')

classes = [
    'Bánh bèo',
    'Bánh bột lọc',
    'Bánh căn',
    'Bánh canh',
    'Bánh chưng',
    'Bánh cuốn',
    'Bánh đúc',
    'Bánh giò',
    'Bánh khọt',
    'Bánh mì',
    'Bánh pía',
    'Bánh tét',
    'Bánh tráng nướng',
    'Bánh xèo',
    'Bún bò Huế',
    'Bún đậu mắm tôm',
    'Bún mắm',
    'Bún riêu',
    'Bún thịt nướng',
    'Cá kho tộ',
    'Canh chua',
    'Cao lầu',
    'Cháo lòng',
    'Cơm tấm',
    'Gỏi cuốn',
    'Hủ tiếu',
    'Mì Quảng',
    'Nem chua',
    'Phở',
    'Xôi xéo'
]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img) / 255
    img = np.expand_dims(img, axis=0)
    return img

def VNFoods():
    st.title(" Vietnamese Food Classification")
    # img_test = preprocess_image('data/30VNFoods/Pho.jpg')
    # # model = load_model('model/base_model_best.hdf5')
    # pred_probs = model.predict(img_test)[0]

    # index = np.argmax(pred_probs)
    # label = classes[index]
    # print(label)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess the image
        img = preprocess_image(uploaded_file)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        pred_probs = model.predict(img)[0]
        # Make prediction
        label = label = classes[np.argmax(pred_probs)]
        st.subheader(f"Dự Đoán: {label}")
