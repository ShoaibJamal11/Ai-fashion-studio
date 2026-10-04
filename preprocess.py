import os
from PIL import Image
from rembg import remove, new_session

# u2netp is ultra-lightweight (~4.7MB) and avoids Cloud RAM / OOM limits
session = new_session("u2netp")

def clean_garment_background(input_path: str, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with Image.open(input_path) as img:
        img_rgb = img.convert("RGBA")
        result = remove(img_rgb, session=session)
        result.save(output_path, "PNG")
    return output_path

if __name__ == "__main__":
    if os.path.exists("test_shirt.jpg"):
        clean_garment_background("test_shirt.jpg", "output/test_cleaned.png")
        print("Preprocess test complete.")