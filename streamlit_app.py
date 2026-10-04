import os
import shutil
import streamlit as st

# ── Bridge Streamlit Cloud secrets → os.environ (must happen before engine imports)
try:
    if "HF_TOKEN" in st.secrets:
        os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
    if "REPLICATE_API_TOKEN" in st.secrets:
        os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]
except Exception:
    pass

from preprocess import clean_garment_background
from vton_engine import run_vton
from video_engine import generate_fashion_reel

st.set_page_config(page_title="AI Fashion Studio", page_icon="👗", layout="centered")

st.title("👗 AI Fashion Studio & Runway Reel Generator")
st.caption("Generate high-fashion virtual try-ons and vertical 9:16 runway reels.")

os.makedirs("output", exist_ok=True)

col1, col2 = st.columns(2)
with col1:
    garment_file = st.file_uploader("1. Garment Photo", type=["jpg", "png", "jpeg"])
with col2:
    model_file = st.file_uploader("2. Model Reference Photo", type=["jpg", "png", "jpeg"])

remove_bg = st.checkbox("Auto-remove garment background", value=True)

if st.button("✨ Generate Photoshoot & Reel", type="primary", use_container_width=True):
    if not garment_file or not model_file:
        st.error("Meharbani karke dono photos (Garment aur Model) upload karein!")
    else:
        with st.status("Processing AI Pipeline...", expanded=True) as status:
            garment_path = os.path.join("output", garment_file.name)
            model_path = os.path.join("output", model_file.name)
            
            with open(garment_path, "wb") as f:
                f.write(garment_file.getbuffer())
            with open(model_path, "wb") as f:
                f.write(model_file.getbuffer())

            active_garment = garment_path
            if remove_bg:
                st.write("🧼 Cleaning garment background...")
                cleaned_path = "output/temp_cleaned.png"
                clean_garment_background(garment_path, cleaned_path)
                active_garment = cleaned_path

            vton_success = False
            tryon_path = None
            st.write("👕 Running Virtual Try-On...")
            try:
                tryon_path = run_vton(active_garment, model_path)
                vton_success = True
            except Exception as e:
                err_msg = str(e)
                if "ZeroGPU" in err_msg or "quota" in err_msg.lower():
                    st.error(
                        "⚠️ Hugging Face ZeroGPU quota exceed ho gaya hai. Meharbani karke Streamlit Cloud settings mein `REPLICATE_API_TOKEN` add karein ya thori der baad koshish karein."
                    )
                else:
                    st.error(f"⚠️ Virtual Try-On mein error aaya: {err_msg}")
                status.update(label="Virtual Try-On Failed", state="error", expanded=True)
            
            video_generated = False
            reel_path = "output/web_reel.mp4"
            if vton_success and tryon_path:
                st.write("🎬 Rendering Runway Video Reel...")
                try:
                    generate_fashion_reel(tryon_path, reel_path)
                    video_generated = True
                except Exception as e:
                    st.warning(f"Video generation space busy tha ya response delay hua: {e}")
                
                status.update(label="Photoshoot Complete!", state="complete", expanded=False)

        if vton_success and tryon_path:
            st.subheader("📸 Generated Try-On Result")
            st.image(tryon_path, use_container_width=True)

            if video_generated and os.path.exists(reel_path):
                st.subheader("🎥 9:16 Fashion Runway Reel")
                st.video(reel_path)
