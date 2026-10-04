import os
import shutil
import inspect
import traceback
import requests
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Defensive filepath extractor (handles all Gradio return shapes)
# ---------------------------------------------------------------------------
def _extract_filepath(data) -> str:
    """
    Recursively extract a valid video filepath from whatever
    gradio_client.predict() or replicate.run() returns.

    Handles:
      - plain string                               → returned as-is
      - dict with 'video', 'path', 'name', 'url'  → recurse into value
      - list / tuple                               → inspect first non-None element
      - Replicate FileOutput (has .url attribute)  → return .url
    """
    # FileOutput object from replicate SDK
    if hasattr(data, "url"):
        return str(data.url)
    if hasattr(data, "read"):          # file-like object
        return str(data)

    if isinstance(data, str):
        return data

    if isinstance(data, dict):
        for key in ("video", "path", "name", "url"):
            if key in data and data[key] is not None:
                candidate = _extract_filepath(data[key])
                if candidate:
                    return candidate
        for key, val in data.items():
            if val is None:
                continue
            try:
                candidate = _extract_filepath(val)
                if candidate:
                    return candidate
            except (ValueError, TypeError):
                pass
        raise ValueError(
            f"Could not find a file path inside dict. "
            f"Keys present: {list(data.keys())}\nFull data: {data}"
        )

    if isinstance(data, (list, tuple)):
        for item in data:
            if item is None:
                continue
            try:
                candidate = _extract_filepath(item)
                if candidate:
                    return candidate
            except (ValueError, TypeError):
                pass
        raise ValueError(f"Could not find a file path in sequence. Items: {data}")

    raise ValueError(
        f"Unsupported type {type(data).__name__} for filepath extraction. "
        f"Value: {data!r}"
    )


# ---------------------------------------------------------------------------
# Engine A: Replicate  (stability-ai/stable-video-diffusion)
# ---------------------------------------------------------------------------
def _generate_via_replicate(image_path: str, output_path: str) -> str:
    """
    Use Replicate's stability-ai/stable-video-diffusion model.
    Requires REPLICATE_API_TOKEN to be set in the environment.

    Returns:
        Absolute path to the saved MP4.
    """
    import replicate  # imported here so the module is optional at load-time

    print("🚀 [Engine A] Using Replicate stable-video-diffusion…")

    # Dynamically resolve latest version — never rely on a stale hash
    try:
        model = replicate.models.get("stability-ai/stable-video-diffusion")
        version_id = model.latest_version.id
        model_ref  = f"stability-ai/stable-video-diffusion:{version_id}"
        print(f"   ℹ️  SVD version: {version_id[:8]}…")
    except Exception as e:
        # Fallback to last-known good version
        version_id = "3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438"
        model_ref  = f"stability-ai/stable-video-diffusion:{version_id}"
        print(f"   ⚠️  Could not fetch latest SVD version ({e}). Using fallback.")

    with open(image_path, "rb") as img_f:
        output = replicate.run(
            model_ref,
            input={
                "input_image":       img_f,
                "motion_bucket_id":  127,     # high motion for runway feel
                "frames_per_second": 6,
                "sizing_strategy":   "maintain_aspect_ratio",
                "video_length":      "25_frames_with_svd_xt",
                "cond_aug":          0.02,
                "decoding_t":        14,
                "seed":              42,
            },
        )

    print(f"   📦 Raw result type : {type(output).__name__}")
    print(f"   📦 Raw result value: {str(output)!r:.200}")

    # Extract URL/path from whatever the SDK returns
    raw_url = _extract_filepath(output)
    print(f"   🌐 Video URL: {raw_url}")

    # Download the MP4 from the Replicate CDN
    print("   💾 Downloading MP4 from Replicate CDN…")
    resp = requests.get(raw_url, timeout=120, stream=True)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"✅ [Engine A] Fashion reel saved → {output_path}  ({size_kb:.1f} KB)")
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Engine B: Hugging Face Space  (multimodalart/stable-video-diffusion)
# ---------------------------------------------------------------------------
def _generate_via_hf_space(image_path: str, output_path: str) -> str:
    """
    Use the public Hugging Face SVD Space as fallback.
    Requires the gradio_client package.

    Returns:
        Absolute path to the saved MP4.
    """
    from gradio_client import Client, handle_file
    from huggingface_hub import login

    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass

    print("⏳ [Engine B] Connecting to HF SVD Space with authenticated token…")

    # Build client kwargs compatible with any gradio_client version
    client_kwargs: dict = {}
    sig    = inspect.signature(Client.__init__)
    params = sig.parameters
    if "token" in params and hf_token:
        client_kwargs["token"] = hf_token
    elif "headers" in params and hf_token:
        client_kwargs["headers"] = {"Authorization": f"Bearer {hf_token}"}

    client = Client("multimodalart/stable-video-diffusion", **client_kwargs)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print("🎬 [Engine B] Rendering runway video (~60–90 s on CPU Space)…")

    # /video signature: (image, seed, randomize_seed, motion_bucket_id, fps_id)
    # Returns: ({"video": local_path, "subtitles": None}, seed_float)
    raw_result = client.predict(
        image=handle_file(image_path),
        seed=42,
        randomize_seed=False,
        motion_bucket_id=127,
        fps_id=6,
        api_name="/video",
    )

    print(f"   📦 Raw result type : {type(raw_result).__name__}")
    print(f"   📦 Raw result value: {raw_result!r:.200}")

    video_path = _extract_filepath(raw_result)
    print(f"   📂 Resolved video path: {video_path}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Extracted path '{video_path}' does not exist on disk. "
            f"Full raw result: {raw_result!r}"
        )

    shutil.copy(video_path, output_path)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"✅ [Engine B] Fashion reel saved → {output_path}  ({size_kb:.1f} KB)")
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Public API — dual-engine with automatic fallback
# ---------------------------------------------------------------------------
def generate_fashion_reel(
    image_path: str,
    output_path: str = "output/fashion_reel.mp4",
) -> str:
    """
    Generate a short fashion runway video from a still image.

    Strategy:
      1. If REPLICATE_API_TOKEN is set  → Engine A (Replicate SVD, dedicated GPU)
      2. Otherwise                      → Engine B (HF Space, free tier)

    Args:
        image_path:   Local path to the input image (e.g. output/result.png).
        output_path:  Destination path for the generated MP4.

    Returns:
        Absolute path to the saved MP4 file.

    Raises:
        RuntimeError if both engines fail, with full error details.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    replicate_token = os.getenv("REPLICATE_API_TOKEN")

    # ── Engine A: Replicate ──────────────────────────────────────────────────
    if replicate_token:
        try:
            return _generate_via_replicate(image_path, output_path)
        except Exception as exc_a:
            print(f"\n⚠️  [Engine A] Replicate failed — falling back to HF Space.")
            print(f"   Error detail: {type(exc_a).__name__}: {exc_a}")
            traceback.print_exc()
    else:
        print("ℹ️  REPLICATE_API_TOKEN not set — skipping Engine A, using HF Space directly.")

    # ── Engine B: Hugging Face Space ─────────────────────────────────────────
    try:
        return _generate_via_hf_space(image_path, output_path)
    except Exception as exc_b:
        err_detail = (
            f"[Engine B] HF Space failed: {type(exc_b).__name__}: {exc_b}\n"
            f"{traceback.format_exc()}"
        )
        print(f"\n❌ {err_detail}")
        raise RuntimeError(
            "Both video engines failed.\n"
            + (f"[Engine A] Replicate: {exc_a}\n" if replicate_token else "")
            + f"[Engine B] HF Space : {exc_b}"
        ) from exc_b
