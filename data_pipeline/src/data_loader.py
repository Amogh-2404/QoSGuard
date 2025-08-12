"""
QoSGuard Data Loader
Handles loading and preprocessing of network flow datasets.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
import structlog

logger = structlog.get_logger()


class DataLoader:
    """Loads and preprocesses network flow data for anomaly detection."""
    
    def __init__(self, data_path: str = "./data_pipeline/data"):
        self.data_path = Path(data_path)
        self.feature_columns = [
            'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
            'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'swin', 'dwin',
            'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
            'sjit', 'djit', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat'
        ]
        self.categorical_features = ['proto', 'service', 'state']
        self.numerical_features = [col for col in self.feature_columns if col not in self.categorical_features]
        
    def load_unsw_nb15_sample(self) -> pd.DataFrame:
        """Load UNSW-NB15 sample dataset."""
        try:
            sample_file = self.data_path / "unsw_nb15_sample.csv"
            
            if not sample_file.exists():
                logger.warning("Sample file not found, creating synthetic data")
                return self._create_synthetic_sample()
            
            df = pd.read_csv(sample_file)
            logger.info("Loaded UNSW-NB15 sample", rows=len(df), columns=len(df.columns))
            return df
            
        except Exception as e:
            logger.error("Failed to load dataset", error=str(e))
            return self._create_synthetic_sample()
    
    def _create_synthetic_sample(self, n_samples: int = 10000) -> pd.DataFrame:
        """Create synthetic network flow data for development/demo."""
        logger.info("Creating synthetic network flow data", samples=n_samples)
        
        np.random.seed(42)  # For reproducibility
        
        # Generate synthetic data
        data = {}
        
        # Duration (seconds)
        data['dur'] = np.random.exponential(scale=10, size=n_samples)
        
        # Protocol distribution
        protocols = ['tcp', 'udp', 'icmp', 'igmp']
        data['proto'] = np.random.choice(protocols, size=n_samples, p=[0.6, 0.3, 0.08, 0.02])
        
        # Service distribution  
        services = ['http', 'https', 'ftp', 'ssh', 'dns', 'smtp', 'pop3', 'imap', 'dhcp', 'ntp', 'other']
        data['service'] = np.random.choice(services, size=n_samples, p=[0.25, 0.2, 0.05, 0.03, 0.1, 0.05, 0.02, 0.02, 0.03, 0.02, 0.23])
        
        # Connection state
        states = ['FIN', 'INT', 'CON', 'ECO', 'RST', 'REQ']
        data['state'] = np.random.choice(states, size=n_samples, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05])
        
        # Packet counts
        data['spkts'] = np.random.poisson(lam=50, size=n_samples)
        data['dpkts'] = np.random.poisson(lam=45, size=n_samples)
        
        # Byte counts (correlated with packet counts)
        data['sbytes'] = data['spkts'] * np.random.normal(1000, 300, n_samples)
        data['dbytes'] = data['dpkts'] * np.random.normal(950, 250, n_samples)
        data['sbytes'] = np.maximum(data['sbytes'], 0)
        data['dbytes'] = np.maximum(data['dbytes'], 0)
        
        # Rate (packets per second)
        data['rate'] = (data['spkts'] + data['dpkts']) / (data['dur'] + 1e-6)
        
        # TTL values
        data['sttl'] = np.random.normal(64, 10, n_samples).astype(int)
        data['dttl'] = np.random.normal(64, 10, n_samples).astype(int)
        data['sttl'] = np.clip(data['sttl'], 1, 255)
        data['dttl'] = np.clip(data['dttl'], 1, 255)
        
        # Load (bits per second)
        data['sload'] = data['sbytes'] * 8 / (data['dur'] + 1e-6)
        data['dload'] = data['dbytes'] * 8 / (data['dur'] + 1e-6)
        
        # Loss and retransmission
        data['sloss'] = np.random.poisson(lam=2, size=n_samples)
        data['dloss'] = np.random.poisson(lam=2, size=n_samples)
        
        # Window sizes
        data['swin'] = np.random.normal(8192, 2000, n_samples).astype(int)
        data['dwin'] = np.random.normal(8192, 2000, n_samples).astype(int)
        data['swin'] = np.maximum(data['swin'], 0)
        data['dwin'] = np.maximum(data['dwin'], 0)
        
        # TCP base sequence numbers
        data['stcpb'] = np.random.randint(0, 2**32, size=n_samples)
        data['dtcpb'] = np.random.randint(0, 2**32, size=n_samples)
        
        # Mean packet sizes
        data['smeansz'] = np.where(data['spkts'] > 0, data['sbytes'] / data['spkts'], 0)
        data['dmeansz'] = np.where(data['dpkts'] > 0, data['dbytes'] / data['dpkts'], 0)
        
        # Transaction depth and response body length
        data['trans_depth'] = np.random.poisson(lam=1, size=n_samples)
        data['res_bdy_len'] = np.random.exponential(scale=500, size=n_samples)
        
        # Jitter and interpacket timing
        data['sjit'] = np.random.exponential(scale=0.01, size=n_samples)
        data['djit'] = np.random.exponential(scale=0.01, size=n_samples)
        data['sintpkt'] = np.random.exponential(scale=0.1, size=n_samples)
        data['dintpkt'] = np.random.exponential(scale=0.1, size=n_samples)
        
        # TCP timing
        data['tcprtt'] = np.random.exponential(scale=0.05, size=n_samples)
        data['synack'] = np.random.exponential(scale=0.01, size=n_samples)
        data['ackdat'] = np.random.exponential(scale=0.005, size=n_samples)
        
        # Generate labels (0 = normal, 1 = anomaly)
        # Make about 15% of samples anomalous
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
        data['label'] = np.zeros(n_samples, dtype=int)
        data['label'][anomaly_indices] = 1
        
        # Attack categories for anomalous samples
        attack_categories = ['DoS', 'DDoS', 'Probe', 'R2L', 'U2R', 'Backdoor', 'Analysis', 'Fuzzers', 'Shellcode']
        attack_labels = ['Normal'] * n_samples
        for idx in anomaly_indices:
            attack_labels[idx] = np.random.choice(attack_categories)
        data['attack_cat'] = attack_labels
        
        # Create anomalous patterns
        for idx in anomaly_indices:
            attack_type = data['attack_cat'][idx]
            
            if attack_type in ['DoS', 'DDoS']:
                # High packet/byte rates
                data['rate'][idx] *= np.random.uniform(5, 20)
                data['spkts'][idx] *= np.random.uniform(10, 50)
                data['sbytes'][idx] *= np.random.uniform(5, 15)
                
            elif attack_type == 'Probe':
                # Port scanning - small packets, many connections
                data['smeansz'][idx] *= 0.1
                data['dmeansz'][idx] *= 0.1
                data['dur'][idx] *= 0.1
                
            elif attack_type in ['R2L', 'U2R']:
                # Remote access - longer duration
                data['dur'][idx] *= np.random.uniform(5, 20)
                data['sbytes'][idx] *= np.random.uniform(2, 5)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Preprocess the dataset for ML training.
        
        Returns:
            Tuple of (preprocessed_df, preprocessing_artifacts)
        """
        try:
            logger.info("Preprocessing data", rows=len(df))
            
            # Make a copy
            df_processed = df.copy()
            artifacts = {}
            
            # Handle missing values
            df_processed = self._handle_missing_values(df_processed)
            
            # Encode categorical features
            df_processed, label_encoders = self._encode_categorical_features(df_processed)
            artifacts['label_encoders'] = label_encoders
            
            # Scale numerical features
            df_processed, scaler = self._scale_numerical_features(df_processed)
            artifacts['scaler'] = scaler
            
            # Feature engineering
            df_processed = self._engineer_features(df_processed)
            
            # Remove outliers
            df_processed = self._remove_outliers(df_processed)
            
            logger.info("Data preprocessing completed", 
                       final_rows=len(df_processed),
                       features=len(self.feature_columns))
            
            return df_processed, artifacts
            
        except Exception as e:
            logger.error("Data preprocessing failed", error=str(e))
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Fill numerical columns with median
        for col in self.numerical_features:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        for col in self.categorical_features:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Encode categorical features."""
        label_encoders = {}
        
        for col in self.categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        
        return df, label_encoders
    
    def _scale_numerical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """Scale numerical features."""
        scaler = StandardScaler()
        
        numerical_cols_present = [col for col in self.numerical_features if col in df.columns]
        if numerical_cols_present:
            df[numerical_cols_present] = scaler.fit_transform(df[numerical_cols_present])
        
        return df, scaler
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional engineered features."""
        # Byte rate ratios
        df['byte_ratio'] = np.where(df['dbytes'] > 0, df['sbytes'] / df['dbytes'], 0)
        df['packet_ratio'] = np.where(df['dpkts'] > 0, df['spkts'] / df['dpkts'], 0)
        
        # Flow efficiency
        total_bytes = df['sbytes'] + df['dbytes']
        total_packets = df['spkts'] + df['dpkts']
        df['bytes_per_packet'] = np.where(total_packets > 0, total_bytes / total_packets, 0)
        
        # Connection activity
        df['total_load'] = df['sload'] + df['dload']
        df['load_ratio'] = np.where(df['dload'] > 0, df['sload'] / df['dload'], 0)
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using z-score method."""
        numerical_cols_present = [col for col in self.numerical_features if col in df.columns]
        
        for col in numerical_cols_present:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores <= z_threshold]
        
        return df
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                  val_size: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, ...]:
        """Split data into train/validation/test sets."""
        # Separate features and labels
        feature_cols = [col for col in self.feature_columns if col in df.columns]
        X = df[feature_cols]
        y = df['label'] if 'label' in df.columns else np.zeros(len(df))
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        logger.info("Data split completed",
                   train_size=len(X_train),
                   val_size=len(X_val),
                   test_size=len(X_test),
                   anomaly_rate_train=y_train.mean(),
                   anomaly_rate_val=y_val.mean(),
                   anomaly_rate_test=y_test.mean())
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def handle_class_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series, 
                              method: str = 'undersample') -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance in training data."""
        logger.info("Handling class imbalance", method=method)
        
        # Combine X and y for resampling
        train_df = X_train.copy()
        train_df['label'] = y_train
        
        # Separate majority and minority classes
        majority_class = train_df[train_df['label'] == 0]
        minority_class = train_df[train_df['label'] == 1]
        
        logger.info("Class distribution before balancing",
                   normal=len(majority_class),
                   anomaly=len(minority_class))
        
        if method == 'undersample':
            # Undersample majority class
            majority_downsampled = resample(
                majority_class,
                replace=False,
                n_samples=len(minority_class) * 2,  # 2:1 ratio
                random_state=42
            )
            balanced_df = pd.concat([majority_downsampled, minority_class])
            
        elif method == 'oversample':
            # Oversample minority class
            minority_upsampled = resample(
                minority_class,
                replace=True,
                n_samples=len(majority_class) // 2,  # 2:1 ratio
                random_state=42
            )
            balanced_df = pd.concat([majority_class, minority_upsampled])
        
        else:
            balanced_df = train_df
        
        # Shuffle the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Separate X and y again
        X_balanced = balanced_df.drop('label', axis=1)
        y_balanced = balanced_df['label']
        
        logger.info("Class distribution after balancing",
                   normal=sum(y_balanced == 0),
                   anomaly=sum(y_balanced == 1))
        
        return X_balanced, y_balanced
