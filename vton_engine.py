import os
import shutil
from gradio_client import Client, handle_file

def run_vton(garment_path: str, model_path: str, category: str = "upper_body") -> str:
    """
    100% Free IDM-VTON execution via Hugging Face Spaces.
    No API token or credit card required.
    """
    print("⏳ Connecting to Hugging Face IDM-VTON Space (Free Tier)...")
    
    # Official IDM-VTON Free Space
    client = Client("yisol/IDM-VTON")
    
    print("⏳ Processing garment warping on cloud GPU (this takes ~30-60s)...")
    
    result = client.predict(
        dict={
            "background": handle_file(model_path),
            "layers": [],
            "composite": None
        },
        garm_img=handle_file(garment_path),
        garment_des="vintage thrift fashion piece, accurate texture, sharp details",
        is_checked=True,        # Auto-masking enabled
        is_checked_crop=False,   # Don't auto-crop
        denoise_steps=30,
        seed=42,
        api_name="/tryon"
    )

    # result tuple return karta hai: (final_image_path, mask_image_path)
    final_image_path = result[0]
    print(f"✅ Try-On Completed successfully!")
    return final_image_path