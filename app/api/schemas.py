"""
QoSGuard API Schemas
Pydantic models for API request/response validation.
"""

from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field


class QoSAction(str, Enum):
    """QoS recommendation actions."""
    PRIORITIZE = "PRIORITIZE"
    RATE_LIMIT = "RATE_LIMIT" 
    DROP_CANDIDATE = "DROP_CANDIDATE"
    INSPECT = "INSPECT"


class FlowFeatures(BaseModel):
    """Network flow features for anomaly detection."""
    # Basic flow features (based on UNSW-NB15 dataset)
    dur: float = Field(..., description="Flow duration")
    proto: str = Field(..., description="Protocol (tcp, udp, icmp, etc.)")
    service: str = Field(..., description="Network service")
    state: str = Field(..., description="Connection state")
    
    # Packet and byte counts
    spkts: int = Field(..., description="Source to destination packets")
    dpkts: int = Field(..., description="Destination to source packets") 
    sbytes: int = Field(..., description="Source to destination bytes")
    dbytes: int = Field(..., description="Destination to source bytes")
    
    # Rate features
    rate: float = Field(default=0.0, description="Packets per second")
    sttl: int = Field(default=0, description="Source to destination TTL")
    dttl: int = Field(default=0, description="Destination to source TTL")
    
    # Load and loss features
    sload: float = Field(default=0.0, description="Source bits per second")
    dload: float = Field(default=0.0, description="Destination bits per second")
    sloss: int = Field(default=0, description="Source packets retransmitted")
    dloss: int = Field(default=0, description="Destination packets retransmitted")
    
    # Window and mean packet size
    swin: int = Field(default=0, description="Source TCP window advertisement")
    dwin: int = Field(default=0, description="Destination TCP window advertisement")
    stcpb: int = Field(default=0, description="Source TCP sequence number")
    dtcpb: int = Field(default=0, description="Destination TCP sequence number")
    smeansz: float = Field(default=0.0, description="Mean source packet size")
    dmeansz: float = Field(default=0.0, description="Mean destination packet size")
    
    # Transaction and response features  
    trans_depth: int = Field(default=0, description="Pipelined depth")
    res_bdy_len: int = Field(default=0, description="Response body length")
    
    # Computed features
    sjit: float = Field(default=0.0, description="Source jitter")
    djit: float = Field(default=0.0, description="Destination jitter")
    sintpkt: float = Field(default=0.0, description="Source interpacket arrival time")
    dintpkt: float = Field(default=0.0, description="Destination interpacket arrival time")
    
    # Connection features
    tcprtt: float = Field(default=0.0, description="TCP round trip time")
    synack: float = Field(default=0.0, description="TCP SYN-ACK time")
    ackdat: float = Field(default=0.0, description="TCP ACK-DAT time")
    
    # Label features for training (not used in inference)
    label: Optional[int] = Field(default=None, description="Ground truth label (0=normal, 1=anomaly)")
    attack_cat: Optional[str] = Field(default=None, description="Attack category")


class PredictionRequest(BaseModel):
    """Single prediction request."""
    flow_id: str = Field(..., description="Unique flow identifier")
    features: FlowFeatures = Field(..., description="Network flow features")
    model_name: Optional[str] = Field(default=None, description="Specific model to use (defaults to ensemble)")
    include_explanation: bool = Field(default=False, description="Include SHAP explanation")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    flows: List[PredictionRequest] = Field(..., description="List of flows to predict")
    model_name: Optional[str] = Field(default=None, description="Model to use for all predictions")


class ModelPrediction(BaseModel):
    """Individual model prediction result."""
    model_name: str
    is_anomaly: bool
    anomaly_score: float
    confidence: float


class PredictionResponse(BaseModel):
    """Prediction response."""
    flow_id: str
    is_anomaly: bool
    anomaly_score: float = Field(..., description="Anomaly score (0-1, higher = more anomalous)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    model_used: str
    qos_recommendation: QoSAction
    processing_time_ms: int
    explanation: Optional[Dict[str, Any]] = Field(default=None, description="SHAP explanation if requested")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    batch_size: int
    successful_predictions: int
    processing_time_ms: int


class QoSThresholds(BaseModel):
    """QoS decision thresholds."""
    prioritize_threshold: float = Field(default=0.9, description="Threshold for PRIORITIZE action")
    rate_limit_threshold: float = Field(default=0.7, description="Threshold for RATE_LIMIT action") 
    drop_threshold: float = Field(default=0.5, description="Threshold for DROP_CANDIDATE action")
    inspect_threshold: float = Field(default=0.3, description="Threshold for INSPECT action")


class QoSPolicy(BaseModel):
    """QoS policy configuration."""
    policy_name: str = Field(default="default", description="Policy name")
    version: str = Field(default="1.0", description="Policy version")
    thresholds: QoSThresholds = Field(default_factory=QoSThresholds)
    enabled_actions: List[QoSAction] = Field(
        default=[QoSAction.PRIORITIZE, QoSAction.RATE_LIMIT, QoSAction.INSPECT],
        description="Enabled QoS actions"
    )
    description: Optional[str] = Field(default=None, description="Policy description")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="Last update timestamp")


class PolicyUpdateRequest(BaseModel):
    """Request to update QoS policy."""
    policy_name: str
    thresholds: Optional[QoSThresholds] = None
    enabled_actions: Optional[List[QoSAction]] = None
    description: Optional[str] = None


class ExplanationRequest(BaseModel):
    """Request for prediction explanation."""
    flow_id: str
    model_name: Optional[str] = Field(default="ensemble")


class FeatureImportance(BaseModel):
    """Feature importance from explanation."""
    feature_name: str
    importance_score: float
    feature_value: Any


class ExplanationResponse(BaseModel):
    """SHAP explanation response."""
    flow_id: str
    model_name: str
    feature_importance: List[FeatureImportance]
    shap_values: List[float]
    base_value: float
    feature_names: List[str]


class ModelInfo(BaseModel):
    """Information about a trained model."""
    name: str
    type: str  # "logistic_regression", "lightgbm", "pytorch_mlp"
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    training_date: str
    model_size_mb: float
    inference_time_ms: float


class MetricsResponse(BaseModel):
    """System metrics response."""
    predictions_total: int
    predictions_per_second: float
    average_latency_ms: float
    model_accuracy: Dict[str, float]
    active_connections: int
    uptime_seconds: int
