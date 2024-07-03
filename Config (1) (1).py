# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python (Local)
#     language: python
#     name: base
# ---

# +
import logging
from langchain_google_vertexai import VertexAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configuration dictionary to store settings for Vertex AI language models
VERTEX_AI_CONFIG = {
    "llm": {
        "model_name": "gemini-1.0-pro-002",
        "temperature": 0.5,
        "max_output_tokens": 8192
    }
}

# Function to create LLM instances based on configurations
def get_vertex_ai_llm(key):
    """
    Retrieves Vertex AI language model instances based on a given configuration key.

    Args:
        key (str): A string key to identify the model configuration.

    Returns:
        VertexAI: A configured instance of VertexAI.

    Raises:
        ValueError: If no configuration is found for the provided key.
    """
    logging.info(f"Retrieving Vertex AI LLM with key: {key}")
    config = VERTEX_AI_CONFIG.get(key, {})
    if not config:
        logging.error(f"No configuration found for key: {key}")
        raise ValueError(f"No configuration found for key: {key}")
    logging.info(f"Configuration found for key: {key}, creating VertexAI instance")
    return VertexAI(**config)

