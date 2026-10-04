import os
import shutil
import inspect
import traceback
import requests
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Engine A: Replicate  (cuuupid/idm-vton)
# ---------------------------------------------------------------------------
def _run_vton_replicate(garment_path: str, model_path: str, category: str = "upper_body") -> str:
    """
    Run Virtual Try-On via Replicate (cuuupid/idm-vton).
    Bypasses Hugging Face ZeroGPU quota and locks completely.
    """
    import replicate

    print("🚀 [Engine A] Running Virtual Try-On via Replicate (cuuupid/idm-vton)…")
    token = os.environ.get("REPLICATE_API_TOKEN", "").strip().strip('"').strip("'")
    client = replicate.Client(api_token=token)

    valid_categories = ["upper_body", "lower_body", "dresses"]
    if category not in valid_categories:
        category = "upper_body"

    # Dynamically resolve latest version or use known good version
    try:
        model = client.models.get("cuuupid/idm-vton")
        version_id = model.latest_version.id
        model_ref = f"cuuupid/idm-vton:{version_id}"
        print(f"   ℹ️  IDM-VTON version: {version_id[:8]}…")
    except Exception as e:
        version_id = "0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985"
        model_ref = f"cuuupid/idm-vton:{version_id}"
        print(f"   ⚠️  Could not fetch latest version ({e}). Using fallback {version_id[:8]}…")

    with open(garment_path, "rb") as garm_f, open(model_path, "rb") as human_f:
        output = client.run(
            model_ref,
            input={
                "garm_img": garm_f,
                "human_img": human_f,
                "garment_des": "vintage thrift fashion piece, accurate texture, sharp details",
                "category": category,
                "seed": 42,
            },
        )

    # Extract output URL
    if isinstance(output, (list, tuple)) and len(output) > 0:
        raw = output[0]
    else:
        raw = output

    result_url = raw.url if hasattr(raw, "url") else str(raw)
    print(f"   🌐 Replicate Try-On Result URL: {result_url}")

    # Save to local file so downstream video engine and Streamlit display it reliably
    os.makedirs("output", exist_ok=True)
    dest_path = os.path.join("output", "vton_result.png")
    if result_url.startswith("http://") or result_url.startswith("https://"):
        print("   💾 Downloading Replicate result to local disk…")
        resp = requests.get(result_url, timeout=120)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        size_kb = os.path.getsize(dest_path) / 1024
        print(f"✅ [Engine A] Try-On saved → {dest_path} ({size_kb:.1f} KB)")
        return os.path.abspath(dest_path)
    elif os.path.exists(result_url):
        shutil.copy(result_url, dest_path)
        return os.path.abspath(dest_path)

    return result_url


# ---------------------------------------------------------------------------
# Engine B: Hugging Face Space  (yisol/IDM-VTON)
# ---------------------------------------------------------------------------
def _run_vton_hf(garment_path: str, model_path: str, category: str = "upper_body") -> str:
    """
    Run Virtual Try-On via Hugging Face Space (yisol/IDM-VTON).
    Authenticated with HF_TOKEN to avoid anonymous IP ZeroGPU quota exhaustion.
    """
    from gradio_client import Client, handle_file

    hf_token = os.environ.get("HF_TOKEN", "").strip().strip('"').strip("'") or None
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass

    print("⏳ [Engine B] Connecting to Hugging Face IDM-VTON Space with authenticated token…")

    sig = inspect.signature(Client.__init__)
    client_kwargs = {}
    if "token" in sig.parameters and hf_token:
        client_kwargs["token"] = hf_token
    elif "hf_token" in sig.parameters and hf_token:
        client_kwargs["hf_token"] = hf_token
    elif "headers" in sig.parameters and hf_token:
        client_kwargs["headers"] = {"Authorization": f"Bearer {hf_token}"}

    try:
        client = Client("yisol/IDM-VTON", **client_kwargs)
    except TypeError:
        client = Client("yisol/IDM-VTON", hf_token=hf_token)

    print("⏳ [Engine B] Processing garment warping on cloud GPU (this takes ~30-60s)…")
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

    final_image_path = result[0] if isinstance(result, (list, tuple)) else result
    os.makedirs("output", exist_ok=True)
    dest_path = os.path.join("output", "vton_result.png")
    if isinstance(final_image_path, str) and os.path.exists(final_image_path):
        shutil.copy(final_image_path, dest_path)
        size_kb = os.path.getsize(dest_path) / 1024
        print(f"✅ [Engine B] Try-On saved → {dest_path} ({size_kb:.1f} KB)")
        return os.path.abspath(dest_path)

    return str(final_image_path)


# ---------------------------------------------------------------------------
# Public API — Dual-Engine Virtual Try-On
# ---------------------------------------------------------------------------
def run_vton(garment_path: str, model_path: str, category: str = "upper_body") -> str:
    """
    Run Virtual Try-On with Dual-Engine strategy:
      1. Primary: Replicate (cuuupid/idm-vton) if REPLICATE_API_TOKEN is present
      2. Fallback: Hugging Face Space (yisol/IDM-VTON) with authenticated HF_TOKEN

    Returns:
        Local path or URL to the generated try-on image.
    """
    replicate_token = os.environ.get("REPLICATE_API_TOKEN", "").strip().strip('"').strip("'")
    err_replicate: str | None = None
    err_hf: str | None = None

    # ── Engine A: Replicate ──────────────────────────────────────────────────
    if replicate_token:
        try:
            return _run_vton_replicate(garment_path, model_path, category)
        except Exception as exc:
            err_replicate = f"{type(exc).__name__}: {exc}"
            print(f"\n⚠️  [Engine A] Replicate VTON failed — falling back to Hugging Face Space.")
            print(f"   Error detail: {err_replicate}")
            traceback.print_exc()
    else:
        print("ℹ️  REPLICATE_API_TOKEN not set — skipping Engine A, using Hugging Face Space directly.")

    # ── Engine B: Hugging Face Space ─────────────────────────────────────────
    try:
        return _run_vton_hf(garment_path, model_path, category)
    except Exception as exc:
        err_hf = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        print(f"\n❌ [Engine B] Hugging Face Space VTON failed:\n{err_hf}")

    raise RuntimeError(
        "Virtual Try-On Failed on all available engines.\n"
        + (f"Replicate: {err_replicate}\n" if err_replicate is not None else "Replicate: not attempted (no token)\n")
        + f"HF Space: {err_hf}"
    )