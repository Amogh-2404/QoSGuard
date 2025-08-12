"""
QoSGuard API Routes
Main API endpoints for predictions, policy management, and monitoring.
"""

from typing import Dict, List, Any, Union
import time
import asyncio

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Histogram, Gauge
import structlog

from app.api.schemas import (
    FlowFeatures, 
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    QoSPolicy,
    PolicyUpdateRequest,
    ExplanationRequest,
    ExplanationResponse
)
from app.models.ml_models import ModelRegistry
from app.core.qos_engine import QoSRecommendationEngine
from app.core.config import get_settings

logger = structlog.get_logger()

# Prometheus metrics
PREDICTION_COUNTER = Counter('qosguard_predictions_total', 'Total predictions made', ['model_type', 'result'])
PREDICTION_LATENCY = Histogram('qosguard_prediction_duration_seconds', 'Prediction latency')
ACTIVE_CONNECTIONS = Gauge('qosguard_active_connections', 'Active WebSocket connections')

router = APIRouter()
settings = get_settings()


async def get_model_registry() -> ModelRegistry:
    """Dependency to get model registry."""
    # This would be injected from app state in a real setup
    return ModelRegistry()


async def get_qos_engine() -> QoSRecommendationEngine:
    """Dependency to get QoS recommendation engine."""
    return QoSRecommendationEngine()


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    model_registry: ModelRegistry = Depends(get_model_registry),
    qos_engine: QoSRecommendationEngine = Depends(get_qos_engine)
) -> PredictionResponse:
    """Make a single anomaly prediction and QoS recommendation."""
    start_time = time.time()
    
    try:
        logger.info("Processing prediction request", flow_id=request.flow_id)
        
        # Get model prediction
        prediction_result = await model_registry.predict(
            features=request.features,
            model_name=request.model_name
        )
        
        # Get QoS recommendation
        qos_action = await qos_engine.recommend_action(
            anomaly_score=prediction_result.anomaly_score,
            features=request.features
        )
        
        # Record metrics
        PREDICTION_COUNTER.labels(
            model_type=request.model_name or "ensemble",
            result="anomaly" if prediction_result.is_anomaly else "normal"
        ).inc()
        
        response = PredictionResponse(
            flow_id=request.flow_id,
            is_anomaly=prediction_result.is_anomaly,
            anomaly_score=prediction_result.anomaly_score,
            confidence=prediction_result.confidence,
            model_used=prediction_result.model_name,
            qos_recommendation=qos_action,
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
        
        # Log prediction for monitoring/auditing
        background_tasks.add_task(
            _log_prediction,
            request.flow_id,
            response
        )
        
        return response
        
    except Exception as e:
        logger.error("Prediction failed", error=str(e), flow_id=request.flow_id)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    finally:
        PREDICTION_LATENCY.observe(time.time() - start_time)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    model_registry: ModelRegistry = Depends(get_model_registry),
    qos_engine: QoSRecommendationEngine = Depends(get_qos_engine)
) -> BatchPredictionResponse:
    """Make batch predictions for multiple flows."""
    start_time = time.time()
    
    try:
        logger.info("Processing batch prediction", batch_size=len(request.flows))
        
        predictions = []
        
        # Process each flow
        for flow_request in request.flows:
            try:
                prediction_result = await model_registry.predict(
                    features=flow_request.features,
                    model_name=request.model_name
                )
                
                qos_action = await qos_engine.recommend_action(
                    anomaly_score=prediction_result.anomaly_score,
                    features=flow_request.features
                )
                
                predictions.append(PredictionResponse(
                    flow_id=flow_request.flow_id,
                    is_anomaly=prediction_result.is_anomaly,
                    anomaly_score=prediction_result.anomaly_score,
                    confidence=prediction_result.confidence,
                    model_used=prediction_result.model_name,
                    qos_recommendation=qos_action,
                    processing_time_ms=int((time.time() - start_time) * 1000)
                ))
                
            except Exception as e:
                logger.warning("Individual prediction failed", 
                             flow_id=flow_request.flow_id, error=str(e))
                continue
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(request.flows),
            successful_predictions=len(predictions),
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
        
    except Exception as e:
        logger.error("Batch prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get("/policy", response_model=QoSPolicy)
async def get_policy(
    qos_engine: QoSRecommendationEngine = Depends(get_qos_engine)
) -> QoSPolicy:
    """Get current QoS policy configuration."""
    try:
        policy = await qos_engine.get_current_policy()
        return policy
    except Exception as e:
        logger.error("Failed to get policy", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get policy: {str(e)}")


@router.put("/policy", response_model=QoSPolicy)
async def update_policy(
    request: PolicyUpdateRequest,
    qos_engine: QoSRecommendationEngine = Depends(get_qos_engine)
) -> QoSPolicy:
    """Update QoS policy configuration."""
    try:
        logger.info("Updating QoS policy", policy_name=request.policy_name)
        
        updated_policy = await qos_engine.update_policy(request)
        
        logger.info("QoS policy updated successfully", policy_name=request.policy_name)
        return updated_policy
        
    except Exception as e:
        logger.error("Failed to update policy", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update policy: {str(e)}")


@router.get("/explain")
async def explain_prediction(
    flow_id: str,
    model_name: str = "ensemble",
    model_registry: ModelRegistry = Depends(get_model_registry)
) -> ExplanationResponse:
    """Get SHAP explanation for a specific prediction."""
    try:
        logger.info("Generating explanation", flow_id=flow_id, model=model_name)
        
        explanation = await model_registry.explain_prediction(flow_id, model_name)
        
        return ExplanationResponse(
            flow_id=flow_id,
            model_name=model_name,
            feature_importance=explanation.feature_importance,
            shap_values=explanation.shap_values,
            base_value=explanation.base_value,
            feature_names=explanation.feature_names
        )
        
    except Exception as e:
        logger.error("Failed to generate explanation", flow_id=flow_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {str(e)}")


@router.get("/models")
async def list_models(
    model_registry: ModelRegistry = Depends(get_model_registry)
) -> Dict[str, Any]:
    """Get information about available models."""
    try:
        models_info = await model_registry.get_models_info()
        return models_info
    except Exception as e:
        logger.error("Failed to get models info", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get models info: {str(e)}")


async def _log_prediction(flow_id: str, response: PredictionResponse) -> None:
    """Background task to log prediction for audit trail."""
    try:
        # In production, this would write to a database or logging system
        logger.info("Prediction logged", 
                   flow_id=flow_id,
                   is_anomaly=response.is_anomaly,
                   anomaly_score=response.anomaly_score,
                   qos_recommendation=response.qos_recommendation)
    except Exception as e:
        logger.error("Failed to log prediction", flow_id=flow_id, error=str(e))
