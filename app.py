import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from rembg import remove
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Face & Text Swapper", layout="wide")
st.title("🖼️ Face & Text Swapper")
st.markdown("Upload image → Remove background → Place new face → Change bottom text")

# File uploads
col1, col2 = st.columns(2)

with col1:
    source_img = st.file_uploader("Upload Original Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    face_img = st.file_uploader("Upload New Face Image", type=["png", "jpg", "jpeg"])

with col2:
    new_text = st.text_input("New Text (Bottom)", placeholder="Enter new text here...")
    text_size = st.slider("Text Size", 20, 120, 60)
    text_color = st.color_picker("Text Color", "#FFFFFF")
    face_scale = st.slider("Face Size Scale (%)", 30, 150, 80)

if source_img and face_img and st.button("🚀 Process Image"):
    with st.spinner("Processing... This may take a few seconds..."):
        try:
            # Load images
            source = Image.open(source_img).convert("RGB")
            face = Image.open(face_img).convert("RGBA")
            
            # Remove background from source
            source_no_bg = remove(source)
            
            # Convert source to RGBA for transparency
            source_no_bg = source_no_bg.convert("RGBA")
            
            # Resize face
            face_width = int(source.width * (face_scale / 100))
            face = face.resize((face_width, int(face_width * face.height / face.width)), Image.Resampling.LANCZOS)
            
            # Simple face placement - center it roughly where a face would be
            x = (source_no_bg.width - face.width) // 2
            y = (source_no_bg.height - face.height) // 3  # Place higher up
            
            # Paste new face onto source
            result_img = source_no_bg.copy()
            result_img.paste(face, (x, y), face)
            
            # Add new text at bottom
            draw = ImageDraw.Draw(result_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", text_size)
            except:
                font = ImageFont.load_default()
            
            # Center text
            bbox = draw.textbbox((0, 0), new_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = (result_img.width - text_width) // 2
            text_y = result_img.height - text_size - 50
            
            # Draw outline + text
            for adj in range(-2, 3):
                for adj2 in range(-2, 3):
                    draw.text((text_x + adj, text_y + adj2), new_text, font=font, fill="black")
            
            draw.text((text_x, text_y), new_text, font=font, fill=text_color)
            
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
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.info("Please upload both images and click Process")
