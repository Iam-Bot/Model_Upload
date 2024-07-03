
"""
File Name: config.py
Author: Denesh A, Vibhish R (Version 1)
Purpose: This script sets up the environment and configuration for Vertex AI language models and Google Cloud services.
Date: 2024-05-20

Examples:
    - Set up environment:
        setup_environment()
    - Retrieve Vertex AI language model instance:
        model = get_vertex_ai_llm('llm')
"""

import os
import logging
from langchain_google_vertexai import VertexAI
from vertexai.preview.generative_models import GenerativeModel
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from google.cloud.sql.connector import Connector
import sqlalchemy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Environment Setup
def setup_environment():
    """
    Configure all necessary environment variables and settings.
    Sets the path for Google application credentials.

    Example:
        setup_environment()
    """
    logging.info("Setting up environment")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"Config/lumen-b-ctl-047-e2aeb24b0ea0.json"
    logging.info("Environment setup complete")

# Configuration dictionary to store settings for Vertex AI language models
VERTEX_AI_CONFIG = {
    "llm": {
        "model_name":"gemini-1.0-pro-002",
        "temperature": 0.5,
        "max_output_tokens": 8192
    },
    "llm_2": {
        "model_name": "gemini-1.5-pro-preview-0409",
        "temperature": 0.3,
        "max_output_tokens": 2048
    }
}

# Call the environment setup function
setup_environment()

# Model name for embeddings and generative model
MODEL_NAME = "textembedding-gecko@002"
GEN_MODEL = GenerativeModel("gemini-1.5-pro-preview-0409")
GEN_MODEL_2 = GenerativeModel("gemini-1.0-pro-002")
CORPUS_NAME = 'projects/844324878551/locations/us-central1/ragCorpora/1664080062313398272'

# Database connection settings
project_id = "lumen-b-ctl-047"
region = "us-central1"
instance_name = "mysql-test"
INSTANCE_CONNECTION_NAME = f"{project_id}:{region}:{instance_name}" # i.e demo-project:us-central1:demo-instance
DB_USER = "test-user"
DB_PASS = "Password@123"
DB_NAME = "mysql_db"

# Initialize Google Cloud SQL Connector
connector = Connector()

# Function to return the database connection object
def getconn():
    """
    Establishes a connection to the Google Cloud SQL database.

    Returns:
        Connection: A database connection object.
    """
    logging.info("Establishing database connection")
    conn = connector.connect(
        INSTANCE_CONNECTION_NAME,
        "pymysql",
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME
    )
    logging.info("Database connection established")
    return conn

# Create SQLAlchemy engine
engine = sqlalchemy.create_engine(
    "mysql+pymysql://",
    creator=getconn,
)

# Create session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base
Base = declarative_base()

#Secret key to sign JWT tokens
SECRET_KEY = "af6e11c710280304f60aa46b69042b467b573ce1c9489dc4c1957e7b25c518eb"

#Algorithm used for JWT tokens
ALGORITHM = "HS256"

#Token expirations for access token
ACCESS_TOKEN_EXPIRE_MINUTES = 30

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

    Example:
        model = get_vertex_ai_llm('llm')
    """
    logging.info(f"Retrieving Vertex AI LLM with key: {key}")
    config = VERTEX_AI_CONFIG.get(key, {})
    if not config:
        logging.error(f"No configuration found for key: {key}")
        raise ValueError(f"No configuration found for key: {key}")
    logging.info(f"Configuration found for key: {key}, creating VertexAI instance")
    return VertexAI(**config)

# Configuration class for managing settings
class AppConfig:
    """
    Configuration class to store application settings.
    """
    CORS_ORIGINS = ["*"]
    BUCKET_NAME = "temp-bucket-chat"
