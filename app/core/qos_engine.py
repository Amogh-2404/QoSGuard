"""
QoSGuard QoS Recommendation Engine
Maps anomaly scores to QoS actions based on configurable policies.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import structlog

from app.api.schemas import QoSAction, QoSPolicy, QoSThresholds, PolicyUpdateRequest, FlowFeatures
from app.core.config import get_settings

logger = structlog.get_logger()


class QoSRecommendationEngine:
    """Engine for making QoS recommendations based on anomaly scores."""
    
    def __init__(self):
        self.settings = get_settings()
        self.current_policy: QoSPolicy = self._load_default_policy()
        self._load_policy_from_file()
    
    def _load_default_policy(self) -> QoSPolicy:
        """Load default QoS policy."""
        return QoSPolicy(
            policy_name="default",
            version="1.0",
            thresholds=QoSThresholds(
                prioritize_threshold=0.9,
                rate_limit_threshold=0.7,
                drop_threshold=0.5,
                inspect_threshold=0.3
            ),
            enabled_actions=[
                QoSAction.PRIORITIZE,
                QoSAction.RATE_LIMIT,
                QoSAction.INSPECT
            ],
            description="Default QoS policy for network anomaly response",
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )
    
    def _load_policy_from_file(self) -> None:
        """Load policy from YAML configuration file."""
        try:
            policy_path = Path(self.settings.policy_config_path)
            
            if policy_path.exists():
                with open(policy_path, 'r') as f:
                    policy_data = yaml.safe_load(f)
                
                # Update current policy with file data
                if policy_data:
                    thresholds = QoSThresholds(**policy_data.get('thresholds', {}))
                    enabled_actions = [
                        QoSAction(action) for action in policy_data.get('enabled_actions', [])
                    ]
                    
                    self.current_policy = QoSPolicy(
                        policy_name=policy_data.get('policy_name', 'default'),
                        version=policy_data.get('version', '1.0'),
                        thresholds=thresholds,
                        enabled_actions=enabled_actions,
                        description=policy_data.get('description'),
                        created_at=policy_data.get('created_at'),
                        updated_at=policy_data.get('updated_at')
                    )
                
                logger.info("Policy loaded from file", 
                           policy=self.current_policy.policy_name,
                           path=str(policy_path))
            
        except Exception as e:
            logger.warning("Failed to load policy from file, using default", error=str(e))
    
    async def recommend_action(self, anomaly_score: float, features: FlowFeatures) -> QoSAction:
        """
        Recommend QoS action based on anomaly score and flow features.
        
        Args:
            anomaly_score: Model anomaly score (0-1, higher = more anomalous)
            features: Network flow features for context
            
        Returns:
            Recommended QoS action
        """
        try:
            thresholds = self.current_policy.thresholds
            enabled_actions = self.current_policy.enabled_actions
            
            # Apply thresholds in priority order
            if (anomaly_score >= thresholds.prioritize_threshold and 
                QoSAction.PRIORITIZE in enabled_actions):
                action = QoSAction.PRIORITIZE
                
            elif (anomaly_score >= thresholds.rate_limit_threshold and 
                  QoSAction.RATE_LIMIT in enabled_actions):
                action = QoSAction.RATE_LIMIT
                
            elif (anomaly_score >= thresholds.drop_threshold and 
                  QoSAction.DROP_CANDIDATE in enabled_actions):
                action = QoSAction.DROP_CANDIDATE
                
            elif (anomaly_score >= thresholds.inspect_threshold and 
                  QoSAction.INSPECT in enabled_actions):
                action = QoSAction.INSPECT
                
            else:
                # Default to inspect for any anomaly
                action = QoSAction.INSPECT
            
            # Apply feature-based adjustments
            action = self._apply_feature_adjustments(action, anomaly_score, features)
            
            logger.debug("QoS action recommended",
                        anomaly_score=anomaly_score,
                        action=action,
                        policy=self.current_policy.policy_name)
            
            return action
            
        except Exception as e:
            logger.error("Failed to recommend QoS action", error=str(e))
            return QoSAction.INSPECT  # Safe default
    
    def _apply_feature_adjustments(self, base_action: QoSAction, 
                                 anomaly_score: float, 
                                 features: FlowFeatures) -> QoSAction:
        """Apply feature-based adjustments to the base recommendation."""
        try:
            # Critical service protection
            if features.service.lower() in ['dns', 'dhcp', 'ntp']:
                if base_action == QoSAction.DROP_CANDIDATE:
                    return QoSAction.RATE_LIMIT  # Don't drop critical services
            
            # High-volume flow handling
            if features.sbytes + features.dbytes > 1000000:  # > 1MB
                if base_action == QoSAction.PRIORITIZE and anomaly_score < 0.95:
                    return QoSAction.RATE_LIMIT  # Rate limit large flows unless very confident
            
            # Protocol-specific adjustments
            if features.proto.upper() == 'ICMP':
                if base_action in [QoSAction.PRIORITIZE, QoSAction.RATE_LIMIT]:
                    return QoSAction.INSPECT  # Be more conservative with ICMP
            
            # Connection state considerations
            if features.state in ['FIN', 'RST']:
                return QoSAction.INSPECT  # Just inspect closing connections
            
            return base_action
            
        except Exception as e:
            logger.error("Feature adjustment failed", error=str(e))
            return base_action
    
    async def get_current_policy(self) -> QoSPolicy:
        """Get the current QoS policy."""
        return self.current_policy
    
    async def update_policy(self, update_request: PolicyUpdateRequest) -> QoSPolicy:
        """
        Update the QoS policy.
        
        Args:
            update_request: Policy update request
            
        Returns:
            Updated policy
        """
        try:
            # Update policy fields
            if update_request.thresholds:
                self.current_policy.thresholds = update_request.thresholds
            
            if update_request.enabled_actions:
                self.current_policy.enabled_actions = update_request.enabled_actions
            
            if update_request.description:
                self.current_policy.description = update_request.description
            
            # Update metadata
            self.current_policy.policy_name = update_request.policy_name
            self.current_policy.updated_at = datetime.utcnow().isoformat()
            
            # Save to file
            await self._save_policy_to_file()
            
            logger.info("QoS policy updated", 
                       policy=self.current_policy.policy_name,
                       thresholds=self.current_policy.thresholds.dict(),
                       actions=self.current_policy.enabled_actions)
            
            return self.current_policy
            
        except Exception as e:
            logger.error("Failed to update policy", error=str(e))
            raise
    
    async def _save_policy_to_file(self) -> None:
        """Save current policy to YAML file."""
        try:
            policy_path = Path(self.settings.policy_config_path)
            policy_path.parent.mkdir(parents=True, exist_ok=True)
            
            policy_dict = {
                'policy_name': self.current_policy.policy_name,
                'version': self.current_policy.version,
                'thresholds': self.current_policy.thresholds.dict(),
                'enabled_actions': [action.value for action in self.current_policy.enabled_actions],
                'description': self.current_policy.description,
                'created_at': self.current_policy.created_at,
                'updated_at': self.current_policy.updated_at
            }
            
            with open(policy_path, 'w') as f:
                yaml.dump(policy_dict, f, default_flow_style=False, indent=2)
            
            logger.info("Policy saved to file", path=str(policy_path))
            
        except Exception as e:
            logger.error("Failed to save policy to file", error=str(e))
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get a summary of the current policy configuration."""
        return {
            'name': self.current_policy.policy_name,
            'version': self.current_policy.version,
            'thresholds': self.current_policy.thresholds.dict(),
            'enabled_actions': [action.value for action in self.current_policy.enabled_actions],
            'action_count': len(self.current_policy.enabled_actions),
            'updated_at': self.current_policy.updated_at
        }
