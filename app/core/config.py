"""
QoSGuard Configuration
Centralized configuration management using Pydantic settings.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "QoSGuard"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # CORS
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        env="ALLOWED_ORIGINS"
    )
    
    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        env="MLFLOW_TRACKING_URI"
    )
    
    # Models
    model_registry_path: str = Field(
        default="./models/registry",
        env="MODEL_REGISTRY_PATH"
    )
    
    # Data
    data_path: str = Field(default="./data_pipeline/data", env="DATA_PATH")
    
    # QoS Policy
    policy_config_path: str = Field(
        default="./policies/policy.yaml",
        env="POLICY_CONFIG_PATH"
    )
    
    # Monitoring
    prometheus_multiproc_dir: str = Field(
        default="/tmp/prometheus_multiproc",
        env="PROMETHEUS_MULTIPROC_DIR"
    )
    
    # AWS (for cloud deployment)
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_s3_bucket: Optional[str] = Field(default=None, env="AWS_S3_BUCKET")
    aws_dynamodb_table: str = Field(
        default="qosguard-policies",
        env="AWS_DYNAMODB_TABLE"
    )
    
    # Model Performance Thresholds
    min_model_accuracy: float = Field(default=0.85, env="MIN_MODEL_ACCURACY")
    max_prediction_latency_ms: int = Field(
        default=300,
        env="MAX_PREDICTION_LATENCY_MS"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
