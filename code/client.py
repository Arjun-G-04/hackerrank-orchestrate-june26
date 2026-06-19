import os
from dotenv import load_dotenv
from loguru import logger

from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

# Load environment variables from .env if present
load_dotenv()


def get_model():
    """
    Factory function to retrieve the configured Gemini model using Google Cloud Provider (Vertex AI).
    """
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash").strip()
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("GCP_LOCATION") or "us-central1"

    logger.info("Initializing GoogleModel with provider: google-cloud (Vertex AI)")
    logger.info(f"Model Name: {model_name}")
    logger.info(f"Project: {project}, Location: {location}")

    if not project:
        logger.warning("GOOGLE_CLOUD_PROJECT environment variable is not set!")

    gcp_provider = GoogleCloudProvider(project=project, location=location)
    return GoogleModel(model_name, provider=gcp_provider)
