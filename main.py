import os
import shutil
import requests
from vton_engine import run_vton

def download_sample_if_needed(url: str, filename: str):
    if not os.path.exists(filename):
        print(f"📥 Downloading test asset: {filename}...")
        res = requests.get(url)
        with open(filename, "wb") as f:
            f.write(res.content)

def main():
    os.makedirs("output", exist_ok=True)

    garment_file = "test_shirt.jpg"
    model_file = "model_reference.jpg"

    # Default sample assets
    download_sample_if_needed(
        "https://raw.githubusercontent.com/yisol/IDM-VTON/main/example/cloth/00057_00.jpg",
        garment_file
    )
    download_sample_if_needed(
        "https://raw.githubusercontent.com/yisol/IDM-VTON/main/example/human/00057_00.jpg",
        model_file
    )

    print("🚀 Starting Free Virtual Try-On Pipeline...")
    output_path = run_vton(
        garment_path=garment_file,
        model_path=model_file,
        category="upper_body"
    )

    destination = "output/result.png"
    if os.path.exists(output_path):
        shutil.copy(output_path, destination)
    else:
        res = requests.get(output_path)
        with open(destination, "wb") as f:
            f.write(res.content)

    print(f"🎉 Success! Generated photoshoot saved at: '{destination}'")

if __name__ == "__main__":
    main()