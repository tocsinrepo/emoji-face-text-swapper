import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from rembg import remove
import insightface
from insightface.app import FaceAnalysis
from io import BytesIO
import os

st.set_page_config(page_title="Face & Text Swapper", layout="wide")
st.title("🖼️ Face & Text Swapper")
st.markdown("Upload image → Remove background → Swap face → Change text")

# Load models
@st.cache_resource
def load_models():
    try:
        face_analyser = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_analyser.prepare(ctx_id=0, det_size=(640, 640))
        swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, providers=['CPUExecutionProvider'])
        return face_analyser, swapper
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None

face_analyser, swapper = load_models()

col1, col2 = st.columns(2)

with col1:
    source_file = st.file_uploader("Upload Original Image", type=["png", "jpg", "jpeg"])
    face_file = st.file_uploader("Upload New Face", type=["png", "jpg", "jpeg"])

with col2:
    new_text = st.text_input("New Bottom Text", placeholder="Enter new text...")
    text_size = st.slider("Text Size", 30, 150, 70)
    text_color = st.color_picker("Text Color", "#FFFFFF")

if source_file and face_file and new_text and st.button("Process Image"):
    if face_analyser is None or swapper is None:
        st.error("Face swap models failed to load. Try on local machine with GPU.")
        st.stop()

    with st.spinner("Processing image..."):
        # Load images
        source = Image.open(source_file).convert("RGB")
        face = Image.open(face_file).convert("RGB")

        # Remove background
        source_no_bg = remove(source)

        # Convert to numpy
        source_np = np.array(source)
        face_np = np.array(face)

        # Face swap
        try:
            faces = face_analyser.get(source_np)
            target_faces = face_analyser.get(face_np)
            
            if len(faces) == 0 or len(target_faces) == 0:
                st.error("Could not detect faces in one of the images.")
                st.stop()

            result = source_np.copy()
            for face_src in faces:
                result = swapper.get(result, face_src, target_faces[0], paste_back=True)

            result_img = Image.fromarray(result)
        except Exception as e:
            st.warning("Face swap failed. Using original image with background removed.")
            result_img = source_no_bg

        # Add text
        draw = ImageDraw.Draw(result_img)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", text_size)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), new_text, font=font)
        text_width = bbox[2] - bbox[0]
        x = (result_img.width - text_width) // 2
        y = result_img.height - text_size - 50

        # Shadow/outline
        for adj in range(-3, 4):
            for adj2 in range(-3, 4):
                draw.text((x + adj, y + adj2), new_text, font=font, fill="black")

        draw.text((x, y), new_text, font=font, fill=text_color)

        # Show result
        st.image(result_img, caption="✅ Final Result", use_column_width=True)

        # Download
        buf = BytesIO()
        result_img.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download PNG",
            buf.getvalue(),
            "swapped_image.png",
            "image/png"
        )
