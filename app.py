import os
import shutil
import gradio as gr
from preprocess import clean_garment_background
from vton_engine import run_vton
from video_engine import generate_fashion_reel

os.makedirs("output", exist_ok=True)

def process_fashion_pipeline(garment_img, model_img, remove_bg):
    if garment_img is None or model_img is None:
        raise gr.Error("Dono photos (Garment aur Model) upload karna zaroori hain!")
    
    # 1. Background cleanup if toggled
    active_garment = garment_img
    if remove_bg:
        cleaned_path = "output/temp_cleaned_garment.png"
        clean_garment_background(garment_img, cleaned_path)
        active_garment = cleaned_path
    
    # 2. Virtual Try-On
    tryon_path = run_vton(
        garment_path=active_garment,
        model_path=model_img,
        category="upper_body"
    )
    
    final_photo_dest = "output/web_result.png"
    if os.path.exists(tryon_path) and tryon_path != final_photo_dest:
        shutil.copy(tryon_path, final_photo_dest)
    
    # 3. Video Reel Generation
    final_video_dest = "output/web_reel.mp4"
    generate_fashion_reel(final_photo_dest, final_video_dest)
    
    return final_photo_dest, final_video_dest

# --- Modern SaaS UI Layout ---
custom_css = """
#main-container {max-width: 1050px; margin: auto; padding: 20px;}
h1 {text-align: center; font-weight: 700; color: #111827;}
p.subtitle {text-align: center; color: #6B7280; margin-bottom: 25px;}
"""

with gr.Blocks(css=custom_css, title="AI Fashion Studio") as demo:
    with gr.Column(elem_id="main-container"):
        gr.Markdown("# 👗 AI Fashion Studio & Runway Reel Generator")
        gr.Markdown("Upload any clothing item and model photo to generate high-fashion e-commerce try-ons and vertical 9:16 reels.", elem_classes=["subtitle"])
        
        with gr.Row():
            with gr.Column():
                garment_input = gr.Image(type="filepath", label="1. Garment Photo (Shirt/Hoodie/Dress)")
                model_input = gr.Image(type="filepath", label="2. Model / Reference Human Photo")
                remove_bg_checkbox = gr.Checkbox(label="Auto-remove garment background (Recommended for flat-lays/hangers)", value=True)
                generate_btn = gr.Button("✨ Generate Photoshoot & Reel", variant="primary", size="lg")
                
            with gr.Column():
                output_image = gr.Image(type="filepath", label="Virtual Try-On Result")
                output_video = gr.Video(label="9:16 Animated Runway Reel", autoplay=True)

        generate_btn.click(
            fn=process_fashion_pipeline,
            inputs=[garment_input, model_input, remove_bg_checkbox],
            outputs=[output_image, output_video]
        )

if __name__ == "__main__":
    # share=True creates a public 72-hour link accessible from mobile phones anywhere
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
