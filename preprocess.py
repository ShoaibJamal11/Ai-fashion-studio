import os
from rembg import remove
from PIL import Image

def clean_garment_background(input_path: str, output_path: str = "cleaned_garment.png") -> str:
    """
    Kapray ke peeche se floor, hanger ya bed ka background hata kar transparent PNG banata hai.
    """
    print(f"✂️ Isolating garment background from: {input_path}...")
    inp = Image.open(input_path)
    output = remove(inp)
    
    # Transparent PNG ko white canvas par merge karna IDM-VTON ke liye best rehta hai
    white_bg = Image.new("RGBA", output.size, "WHITE")
    white_bg.paste(output, (0, 0), output)
    final_img = white_bg.convert("RGB")
    
    final_img.save(output_path, "JPEG")
    print(f"✅ Cleaned garment ready: {output_path}")
    return output_path