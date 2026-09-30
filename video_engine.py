import os
import shutil
import inspect
from dotenv import load_dotenv
from gradio_client import Client, handle_file
from huggingface_hub import login

load_dotenv()


# ---------------------------------------------------------------------------
# Defensive filepath extractor
# ---------------------------------------------------------------------------
def _extract_filepath(data) -> str:
    """
    Recursively extract a valid video/image filepath from whatever
    gradio_client.predict() returns.

    The SVD Space /video endpoint returns:
        ({"video": "/tmp/gradio/.../output.mp4", "subtitles": None}, seed_float)

    This helper handles that case plus all other common Gradio response shapes:
      - plain string          → returned as-is
      - dict with 'video'     → recurse into value          (SVD /video endpoint)
      - dict with 'path'      → return value directly
      - dict with 'name'      → return value directly
      - dict with 'url'       → return value directly
      - list / tuple          → inspect first non-None element
      - nested combinations   → handled via recursion

    Raises a descriptive ValueError if no valid path can be found.
    """
    # ── String ──────────────────────────────────────────────────────────────
    if isinstance(data, str):
        VIDEO_EXTS = (".mp4", ".webm", ".avi", ".mov", ".mkv",
                      ".jpg", ".jpeg", ".png", ".gif", ".webp")
        if os.path.exists(data) or data.lower().endswith(VIDEO_EXTS):
            return data
        # Could still be a temp path that exists later; return anyway
        return data

    # ── Dict ─────────────────────────────────────────────────────────────────
    if isinstance(data, dict):
        # Priority key order matches SVD and common Gradio component schemas
        for key in ("video", "path", "name", "url"):
            if key in data and data[key] is not None:
                candidate = _extract_filepath(data[key])
                if candidate:
                    return candidate

        # Last resort: inspect every value in the dict
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
            f"Could not find a file path inside dict. Keys present: {list(data.keys())}\n"
            f"Full data: {data}"
        )

    # ── List / Tuple ─────────────────────────────────────────────────────────
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
        raise ValueError(
            f"Could not find a file path in sequence. Items: {data}"
        )

    raise ValueError(
        f"Unsupported type {type(data).__name__} for filepath extraction. Value: {data!r}"
    )


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------
def generate_fashion_reel(
    image_path: str,
    output_path: str = "output/fashion_reel.mp4",
    motion_bucket_id: int = 127,
    fps_id: int = 6,
) -> str:
    """
    Generate a short fashion runway video from a still image using
    Stable Video Diffusion (multimodalart/stable-video-diffusion on HF Spaces).

    Args:
        image_path:        Local path to the input image (e.g. output/result.png).
        output_path:       Destination path for the generated MP4.
        motion_bucket_id:  SVD motion intensity (0–255, default 127).
        fps_id:            Frames-per-second ID (default 6 → ~24 fps output).

    Returns:
        Absolute path to the saved MP4 file.
    """
    hf_token = os.getenv("HF_TOKEN")

    # ── HuggingFace login (best-effort) ──────────────────────────────────────
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass

    print("⏳ Connecting to SVD Space with authenticated HuggingFace token…")

    # ── Build Client — use 'token' (correct param for gradio_client ≥ 1.x) ──
    client_kwargs: dict = {}
    sig = inspect.signature(Client.__init__)
    params = sig.parameters

    if "token" in params and hf_token:
        client_kwargs["token"] = hf_token
    elif "headers" in params and hf_token:
        # Older gradio_client builds used headers
        client_kwargs["headers"] = {"Authorization": f"Bearer {hf_token}"}

    client = Client("multimodalart/stable-video-diffusion", **client_kwargs)

    # ── Ensure output directory exists ────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"🎬 Rendering runway video (motion={motion_bucket_id}, fps_id={fps_id})…")
    print("   ⏱  This takes ~60–90 seconds on CPU Spaces — please wait…")

    # ── Call the /video endpoint ──────────────────────────────────────────────
    # Signature: predict(image, seed, randomize_seed, motion_bucket_id, fps_id)
    # Returns:   ({"video": filepath, "subtitles": None}, seed_float)
    raw_result = client.predict(
        image=handle_file(image_path),
        seed=42,
        randomize_seed=False,
        motion_bucket_id=motion_bucket_id,
        fps_id=fps_id,
        api_name="/video",
    )

    print(f"   📦 Raw result type : {type(raw_result).__name__}")
    print(f"   📦 Raw result value: {raw_result!r:.200}")

    # ── Extract the file path defensively ────────────────────────────────────
    video_path = _extract_filepath(raw_result)
    print(f"   📂 Resolved video path: {video_path}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Extracted path '{video_path}' does not exist on disk. "
            f"Full raw result: {raw_result!r}"
        )

    # ── Copy to destination ───────────────────────────────────────────────────
    shutil.copy(video_path, output_path)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"✅ Fashion reel saved → {output_path}  ({size_kb:.1f} KB)")
    return os.path.abspath(output_path)
