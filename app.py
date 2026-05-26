import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from rembg import remove
import insightface
from insightface.app import FaceAnalysis
import os
from io import BytesIO

st.set_page_config(page_title="Face & Text Swapper", layout="wide")
st.title("🖼️ Face & Text Swapper")
st.markdown("Upload image → Remove background → Swap face → Change text")

# Load face swapper
@st.cache_resource
def load_face_swapper():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx')
    return app, swapper

try:
    face_analyser, swapper = load_face_swapper()
except:
    st.error("Face swap model not found. Please download inswapper_128.onnx")
    st.stop()

# File uploads
col1, col2 = st.columns(2)

with col1:
    source_img = st.file_uploader("Upload Original Image (PNG)", type=["png", "jpg", "jpeg"])
    face_img = st.file_uploader("Upload New Face Image", type=["png", "jpg", "jpeg"])

with col2:
    new_text = st.text_input("New Text (Bottom)", placeholder="Enter new text here...")
    text_size = st.slider("Text Size", 20, 120, 60)
    text_color = st.color_picker("Text Color", "#FFFFFF")

if source_img and face_img and st.button("🚀 Process Image"):
    with st.spinner("Processing..."):
        # Load images
        source = Image.open(source_img).convert("RGB")
        face = Image.open(face_img).convert("RGB")
        
        # Remove background
        source_no_bg = remove(source)
        
        # Convert to numpy
        source_np = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
        face_np = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
        
        # Face detection and swap
        faces = face_analyser.get(source_np)
        face_target = face_analyser.get(face_np)[0]
        
        if len(faces) > 0:
            swapped = source_np.copy()
            for face_src in faces:
                swapped = swapper.get(swapped, face_src, face_target, paste_back=True)
            
            # Convert back to PIL
            result_img = Image.fromarray(cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB))
        else:
            st.error("No face detected in source image")
            st.stop()

        # Add new text
        draw = ImageDraw.Draw(result_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", text_size)
        except:
            font = ImageFont.load_default()

        # Center text at bottom
        bbox = draw.textbbox((0, 0), new_text, font=font)
        text_width = bbox[2] - bbox[0]
        x = (result_img.width - text_width) // 2
        y = result_img.height - text_size - 40

        # Draw text with outline for better visibility
        draw.text((x-2, y-2), new_text, font=font, fill="black")
        draw.text((x+2, y-2), new_text, font=font, fill="black")
        draw.text((x-2, y+2), new_text, font=font, fill="black")
        draw.text((x+2, y+2), new_text, font=font, fill="black")
        draw.text((x, y), new_text, font=font, fill=text_color)

        # Display result
        st.image(result_img, caption="Final Result", use_column_width=True)

        # Download button
        buf = BytesIO()
        result_img.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Final Image",
            data=buf.getvalue(),
            file_name="swapped_image.png",
            mime="image/png"
        )
