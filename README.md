# Face & Text Swapper

A Streamlit app to:
- Remove background from uploaded image
- Swap face with new face
- Replace bottom text

## Setup

1. Clone the repo
2. Download `inswapper_128.onnx` from [InsightFace](https://github.com/deepinsight/insightface/releases)  
   Place it in `~/.insightface/models/inswapper_128.onnx`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run:
   ```bash
   streamlit run app.py
   ```

Made with ❤️ using Streamlit + InsightFace + rembg