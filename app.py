import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="AI Image Captioner",
    page_icon="📸",
    layout="centered"
)

st.title("📸 AI Image Checker")
st.write("Upload an image and let the AI describe it for you!")

# --- Load the Pre-trained Model ---
# Use st.cache_resource to load the model only once across sessions
# This is crucial for performance as model loading is slow
@st.cache_resource
def load_model():
    """Loads the BLIP image captioning model and processor."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

# Load the model and processor
processor, model = load_model()

# --- Prediction Function ---
def get_caption(image: Image.Image) -> str:
    """Generates what is in this image."""
    # Unconditional image captioning
    inputs = processor(image, return_tensors="pt")

    # Generate caption
    # Use 'max_new_tokens' to control the length of the caption
    out = model.generate(**inputs, max_new_tokens=50)

    # Decode the generated tokens to text
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# --- Streamlit UI Layout ---

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read the image file
        image = Image.open(uploaded_file).convert("RGB") # Ensure image is in RGB format

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("") # Add a bit of space

        # Generate and display the caption
        with st.spinner("Generating Image Idea..."):
            caption = get_caption(image)
            st.subheader("AI Description:")
            st.success(caption)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("Please ensure the uploaded file is a valid image.")

else:
    st.info("Please upload an image file to get a description.")

st.markdown("---")
st.markdown("Built with ❤️ By Syed Afseh Ehsani")