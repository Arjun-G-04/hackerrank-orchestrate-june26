from pathlib import Path
import pandas as pd
from loguru import logger
from pydantic_ai import BinaryContent

# Root path resolution
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"

# Caching DataFrames
_user_history_df: pd.DataFrame | None = None
_evidence_req_df: pd.DataFrame | None = None
_user_history_cache: dict = {}


def load_context_data():
    """
    Load CSV dataset files into memory for fast lookups.
    """
    global _user_history_df, _evidence_req_df, _user_history_cache

    # 1. Load User History
    user_history_path = DATASET_DIR / "user_history.csv"
    if user_history_path.exists():
        logger.info(f"Loading user history from {user_history_path}")
        _user_history_df = pd.read_csv(user_history_path)
        # Convert to dictionary cache for O(1) lookups
        for _, row in _user_history_df.iterrows():
            u_id = str(row["user_id"]).strip()
            _user_history_cache[u_id] = {
                "user_id": u_id,
                "past_claim_count": int(row["past_claim_count"]),
                "accept_claim": int(row["accept_claim"]),
                "manual_review_claim": int(row["manual_review_claim"]),
                "rejected_claim": int(row["rejected_claim"]),
                "last_90_days_claim_count": int(row["last_90_days_claim_count"]),
                "history_flags": str(row["history_flags"]).strip(),
                "history_summary": str(row["history_summary"]).strip(),
            }
    else:
        logger.warning(f"User history file not found at {user_history_path}")

    # 2. Load Evidence Requirements
    evidence_req_path = DATASET_DIR / "evidence_requirements.csv"
    if evidence_req_path.exists():
        logger.info(f"Loading evidence requirements from {evidence_req_path}")
        _evidence_req_df = pd.read_csv(evidence_req_path)
    else:
        logger.warning(f"Evidence requirements file not found at {evidence_req_path}")


def get_user_history(user_id: str) -> dict:
    """
    Look up user history in O(1) time.
    Returns a dictionary of risk history, or default blank history if not found.
    """
    u_id_clean = str(user_id).strip()
    if not _user_history_cache:
        load_context_data()

    if u_id_clean in _user_history_cache:
        return _user_history_cache[u_id_clean]

    # Default fallback for users with no history
    return {
        "user_id": u_id_clean,
        "past_claim_count": 0,
        "accept_claim": 0,
        "manual_review_claim": 0,
        "rejected_claim": 0,
        "last_90_days_claim_count": 0,
        "history_flags": "none",
        "history_summary": "No prior claim history found",
    }


def get_evidence_requirements(claim_object: str) -> list[dict]:
    """
    Get all evidence requirements applicable to a specific claim_object.
    Includes general requirements (applies to 'all').
    """
    if _evidence_req_df is None:
        load_context_data()

    if _evidence_req_df is None:
        return []

    # Filter for requirements matching the object or 'all'
    filtered = _evidence_req_df[
        (
            _evidence_req_df["claim_object"].str.strip().str.lower()
            == claim_object.strip().lower()
        )
        | (_evidence_req_df["claim_object"].str.strip().str.lower() == "all")
    ]

    reqs = []
    for _, row in filtered.iterrows():
        reqs.append(
            {
                "requirement_id": str(row["requirement_id"]).strip(),
                "claim_object": str(row["claim_object"]).strip(),
                "applies_to": str(row["applies_to"]).strip(),
                "minimum_image_evidence": str(row["minimum_image_evidence"]).strip(),
            }
        )
    return reqs


def load_images_binary(image_paths_str: str) -> list[tuple[str, bytes, str]]:
    """
    Parse a semicolon-separated list of image paths, load their binary contents,
    and return list of tuples: (image_id, data_bytes, media_type)
    """
    if not image_paths_str or pd.isna(image_paths_str):
        return []

    parts = [p.strip() for p in str(image_paths_str).split(";") if p.strip()]
    results = []

    for part in parts:
        # Construct absolute path
        full_path = DATASET_DIR / part
        if not full_path.exists():
            logger.error(f"Image file not found at: {full_path}")
            continue

        # Get image ID (filename without extension)
        image_id = full_path.stem

        # Read file bytes
        with open(full_path, "rb") as f:
            data = f.read()

        # Determine media type based on extension
        suffix = full_path.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            media_type = "image/jpeg"
        elif suffix == ".png":
            media_type = "image/png"
        elif suffix == ".webp":
            media_type = "image/webp"
        else:
            media_type = "application/octet-stream"

        results.append((image_id, data, media_type))

    return results


def prepare_multimodal_inputs(image_paths_str: str) -> list[BinaryContent]:
    """
    Prepares list of Pydantic AI BinaryContent objects for the LLM call.
    """
    binary_images = load_images_binary(image_paths_str)
    contents = []
    for image_id, data, media_type in binary_images:
        contents.append(
            BinaryContent(data=data, media_type=media_type, identifier=image_id)
        )
    return contents
