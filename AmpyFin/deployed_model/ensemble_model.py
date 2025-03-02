# ensemble_model.py
import pandas as pd
import numpy as np
import os
import logging
import joblib
import platform
import json
import traceback
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from collections import defaultdict

# FORCE ENABLE DETAILED LOGGING - Using ERROR level to ensure visibility
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # Force logger to ERROR level

# Machine learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, fbeta_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Print a very obvious startup message
print("\n" + "!"*80)
print("🚨 ENSEMBLE MODEL RELOAD TRIGGERED 🚨")
print("!"*80 + "\n")

# Deep learning imports with improved handling
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    
    # Access components through the keras object
    Sequential = keras.Sequential
    Dense = keras.layers.Dense
    LSTM = keras.layers.LSTM
    GRU = keras.layers.GRU
    Bidirectional = keras.layers.Bidirectional
    Conv1D = keras.layers.Conv1D
    Dropout = keras.layers.Dropout
    BatchNormalization = keras.layers.BatchNormalization
    Attention = keras.layers.Attention
    LayerNormalization = keras.layers.LayerNormalization
    Add = keras.layers.Add
    Adam = keras.optimizers.Adam
    EarlyStopping = keras.callbacks.EarlyStopping
    ModelCheckpoint = keras.callbacks.ModelCheckpoint
    ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
    TensorBoard = keras.callbacks.TensorBoard
    load_model = keras.models.load_model
    save_model = keras.models.save_model
    
    # Enable mixed precision for faster training on compatible hardware
    try:
        mixed_precision = tf.keras.mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled with policy: mixed_float16")
    except:
        logger.info("Mixed precision not enabled")
    
    TENSORFLOW_AVAILABLE = True
    TF_VERSION = tf.__version__
    logger.info(f"TensorFlow {TF_VERSION} successfully imported")
except ImportError as e:
    # More helpful error message
    if platform.processor() == 'arm':
        logger.warning("TensorFlow not found. For Apple Silicon Macs, install with: pip install tensorflow-macos tensorflow-metal")
    else:
        logger.warning("TensorFlow not found. Install with: pip install tensorflow")
    logger.info(f"Deep learning features will be disabled. Error: {e}")


class EnsembleModelManager:
    """
    Advanced ensemble model that combines multiple ML models using stacking approach.
    
    Features:
    - Multiple model types (tree-based, neural networks, and traditional ML)
    - Time series cross-validation
    - Feature importance tracking
    - Automatic hyperparameter tuning
    - Robust error handling
    - Detailed performance metrics
    - Enhanced LSTM architecture with attention
    """
    
    def __init__(self, model_dir="./ensemble_models", n_regimes=3):
        """
        Initialize the ensemble model manager.
        
        Args:
            model_dir: Directory to save/load model files
            n_regimes: Number of market regimes to consider (for regime-specific models)
        """
        print("🔄 INIT: EnsembleModelManager initialization started 🔄")
        logger.error("🔄 INIT: EnsembleModelManager initialization started 🔄")
        self.model_dir = model_dir
        self.n_regimes = n_regimes
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Sub-models in the ensemble
        self.models = {}
        self.scalers = {}
        self.feature_subsets = {}
        
        # Meta-learner model
        self.meta_model = None
        self.meta_scaler = None
        self.meta_feature_names = []  # Store expected meta feature names
        
        # Track feature importance across all models
        self.feature_importance = {}
        
        # Feature columns information
        self.all_feature_cols = []
        
        # Sub-model performance metrics
        self.model_metrics = {}
        
        # Track trained epochs for neural network models
        self.model_epochs = {}
        
        # Regime-specific models
        self.regime_specific_models = {}
        
        # LSTM model (TensorFlow required)
        self.lstm_model = None
        self.lstm_scaler = None
        self.lstm_sequence_length = 10  # Default sequence length for LSTM
        self.lstm_feature_cols = []  # Store feature columns used by LSTM
        
        # Configuration for model training
        self.config = {
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 5,
            'hyperparameter_tuning': False,
            'early_stopping_patience': 10,
            'lstm_batch_size': 32,
            'lstm_epochs': 50,
            'meta_learner_type': 'xgboost',  # 'xgboost', 'lightgbm', 'randomforest', 'voting'
            'use_attention': True,           # Use attention mechanism in LSTM
            'use_feature_selection': True,   # Perform automatic feature selection
        }
        
        # Load models if they exist
        self.load_models()
        print("🔄 INIT: EnsembleModelManager initialization completed 🔄")
        logger.error("🔄 INIT: EnsembleModelManager initialization completed 🔄")
    
    def debug_meta_features(self):
        """
        Extensive debugging method to diagnose meta-features and model issues.
        Call this before making predictions to identify potential problems.
        """
        print("=" * 50)
        print("🔍 ENSEMBLE MODEL DEBUG INFORMATION 🔍")
        print("=" * 50)
        logger.error("=" * 50)
        logger.error("🔍 ENSEMBLE MODEL DEBUG INFORMATION 🔍")
        logger.error("=" * 50)
        
        # Check meta-model
        print("Meta-model status:")
        logger.error("Meta-model status:")
        print(f"  Meta-model exists: {self.meta_model is not None}")
        logger.error(f"  Meta-model exists: {self.meta_model is not None}")
        print(f"  Meta-scaler exists: {self.meta_scaler is not None}")
        logger.error(f"  Meta-scaler exists: {self.meta_scaler is not None}")
        if hasattr(self.meta_model, 'feature_names_in_'):
            print(f"  Meta-model feature_names_in_: {self.meta_model.feature_names_in_}")
            logger.error(f"  Meta-model feature_names_in_: {self.meta_model.feature_names_in_}")
        
        # Check meta feature names
        print("Meta feature names:")
        logger.error("Meta feature names:")
        print(f"  self.meta_feature_names: {self.meta_feature_names}")
        logger.error(f"  self.meta_feature_names: {self.meta_feature_names}")
        
        # Check all base models
        print("Base models:")
        logger.error("Base models:")
        for model_name, model in self.models.items():
            features = self.feature_subsets.get(model_name, [])
            print(f"  {model_name}: {len(features)} features")
            logger.error(f"  {model_name}: {len(features)} features")
            
        # Check LSTM model
        print("LSTM model status:")
        logger.error("LSTM model status:")
        print(f"  LSTM model exists: {self.lstm_model is not None}")
        logger.error(f"  LSTM model exists: {self.lstm_model is not None}")
        print(f"  LSTM scaler exists: {self.lstm_scaler is not None}")
        logger.error(f"  LSTM scaler exists: {self.lstm_scaler is not None}")
        print(f"  LSTM feature columns: {len(self.lstm_feature_cols)} features")
        logger.error(f"  LSTM feature columns: {len(self.lstm_feature_cols)} features")
        
        # Generate meta features for a sample prediction
        print("Generating sample meta features:")
        logger.error("Generating sample meta features:")
        # Create a tiny sample DataFrame with the expected features
        if self.all_feature_cols:
            sample_data = {col: [0.0] for col in self.all_feature_cols}
            sample_df = pd.DataFrame(sample_data)
            model_predictions = {}
            
            # Get predictions from each model for the sample
            for model_name, model in self.models.items():
                features = self.feature_subsets.get(model_name, [])
                if features:
                    model_predictions[f"{model_name}_prob"] = 0.5
            
            # Always add lstm_prob
            model_predictions['lstm_prob'] = 0.5
            
            # Log meta features
            print(f"  Sample meta features: {list(model_predictions.keys())}")
            logger.error(f"  Sample meta features: {list(model_predictions.keys())}")
            
            # Check if the expected meta feature columns match what would be generated
            if self.meta_feature_names:
                missing = [f for f in self.meta_feature_names if f not in model_predictions]
                extra = [f for f in model_predictions if f not in self.meta_feature_names]
                
                if missing:
                    print(f"  🚨 CRITICAL: Missing expected meta features: {missing}")
                    logger.error(f"  🚨 CRITICAL: Missing expected meta features: {missing}")
                if extra:
                    print(f"  ⚠️ WARNING: Extra meta features not in model: {extra}")
                    logger.error(f"  ⚠️ WARNING: Extra meta features not in model: {extra}")
            
        print("=" * 50)
        logger.error("=" * 50)
        
        print("Debug meta-features completed! Check logs for details.")
    
    def load_models(self):
        """Load all models if they exist."""
        try:
            # DEBUGGING
            print("🔄 LOAD: load_models called 🔄")
            logger.error("🔄 LOAD: load_models called 🔄")
            print(f"🔄 LOAD: Looking for models in: {self.model_dir}")
            logger.error(f"🔄 LOAD: Looking for models in: {self.model_dir}")
            
            # Check if directory exists
            if not os.path.exists(self.model_dir):
                print(f"🔄 LOAD: Model directory {self.model_dir} not found. Will create new models.")
                logger.error(f"🔄 LOAD: Model directory {self.model_dir} not found. Will create new models.")
                return False
                
            # Check for model registry file
            registry_path = os.path.join(self.model_dir, "model_registry.json")
            if not os.path.exists(registry_path):
                print("🔄 LOAD: Model registry not found. Will create new models.")
                logger.error("🔄 LOAD: Model registry not found. Will create new models.")
                return False
                
            # Load model registry using json for better compatibility
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            print(f"🔄 LOAD: Successfully loaded registry from {registry_path}")
            logger.error(f"🔄 LOAD: Successfully loaded registry from {registry_path}")
            
            # Load configuration if available
            if 'config' in registry:
                self.config.update(registry['config'])
            
            # Load feature information
            if 'all_feature_cols' in registry:
                self.all_feature_cols = registry['all_feature_cols']
                print(f"🔄 LOAD: Loaded {len(self.all_feature_cols)} feature columns")
                logger.error(f"🔄 LOAD: Loaded {len(self.all_feature_cols)} feature columns")
            
            # Load meta feature names if available
            if 'meta_feature_names' in registry:
                self.meta_feature_names = registry['meta_feature_names']
                print(f"🔄 LOAD: Loaded meta_feature_names: {self.meta_feature_names}")
                logger.error(f"🔄 LOAD: Loaded meta_feature_names: {self.meta_feature_names}")
            
            # Load each sub-model based on registry
            for model_name, model_info in registry.get('models', {}).items():
                model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
                scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.joblib")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[model_name] = joblib.load(model_path)
                    self.scalers[model_name] = joblib.load(scaler_path)
                    self.feature_subsets[model_name] = model_info.get('features', [])
                    print(f"🔄 LOAD: Loaded model {model_name}")
                    logger.error(f"🔄 LOAD: Loaded model {model_name}")
            
            # Load meta-model if it exists
            meta_model_path = os.path.join(self.model_dir, "meta_model.joblib")
            meta_scaler_path = os.path.join(self.model_dir, "meta_scaler.joblib")
            
            if os.path.exists(meta_model_path) and os.path.exists(meta_scaler_path):
                self.meta_model = joblib.load(meta_model_path)
                self.meta_scaler = joblib.load(meta_scaler_path)
                print("🔄 LOAD: Loaded meta-model")
                logger.error("🔄 LOAD: Loaded meta-model")
                
                # Try to access feature names
                if hasattr(self.meta_model, 'feature_names_in_'):
                    print(f"🔄 LOAD: Meta-model feature_names_in_: {self.meta_model.feature_names_in_}")
                    logger.error(f"🔄 LOAD: Meta-model feature_names_in_: {self.meta_model.feature_names_in_}")
            
            # Load feature importance if available
            importance_path = os.path.join(self.model_dir, "feature_importance.joblib")
            if os.path.exists(importance_path):
                self.feature_importance = joblib.load(importance_path)
            
            # Load model metrics if available
            metrics_path = os.path.join(self.model_dir, "model_metrics.joblib")
            if os.path.exists(metrics_path):
                self.model_metrics = joblib.load(metrics_path)
            
            # Load LSTM model if TensorFlow is available
            if TENSORFLOW_AVAILABLE:
                # Try multiple possible LSTM model paths
                lstm_paths = [
                    os.path.join(self.model_dir, "lstm_model.keras"),
                    os.path.join(self.model_dir, "lstm_model.h5")
                ]
                
                # Track if LSTM model was successfully loaded
                lstm_model_loaded = False
                for lstm_path in lstm_paths:
                    if os.path.exists(lstm_path):
                        try:
                            self.lstm_model = load_model(lstm_path)
                            print(f"🔄 LOAD: Loaded LSTM model from {lstm_path}")
                            logger.error(f"🔄 LOAD: Loaded LSTM model from {lstm_path}")
                            lstm_model_loaded = True
                            break
                        except Exception as e:
                            print(f"🔄 LOAD: Error loading LSTM model from {lstm_path}: {e}")
                            logger.error(f"🔄 LOAD: Error loading LSTM model from {lstm_path}: {e}")
                
                # Load LSTM scaler
                lstm_scaler_path = os.path.join(self.model_dir, "lstm_scaler.joblib")
                if os.path.exists(lstm_scaler_path):
                    self.lstm_scaler = joblib.load(lstm_scaler_path)
                    print("🔄 LOAD: Loaded LSTM scaler")
                    logger.error("🔄 LOAD: Loaded LSTM scaler")
                
                # Load LSTM feature columns
                if 'lstm_feature_cols' in registry:
                    self.lstm_feature_cols = registry['lstm_feature_cols']
                
                # Load LSTM sequence length
                if 'lstm_sequence_length' in registry:
                    self.lstm_sequence_length = registry['lstm_sequence_length']
            
            print(f"🔄 LOAD: Successfully loaded {len(self.models)} models and meta-model")
            logger.error(f"🔄 LOAD: Successfully loaded {len(self.models)} models and meta-model")
            return True
            
        except Exception as e:
            print(f"🔄 LOAD ERROR: Error loading models: {e}")
            print(f"🔄 LOAD ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"🔄 LOAD ERROR: Error loading models: {e}")
            logger.error(f"🔄 LOAD ERROR TRACEBACK:\n{traceback.format_exc()}")
            # Reset to empty state
            self.models = {}
            self.scalers = {}
            self.feature_subsets = {}
            self.meta_model = None
            self.meta_scaler = None
            return False
    
    def save_models(self):
        """Save all models and related artifacts."""
        try:
            print("💾 SAVE: save_models called 💾")
            logger.error("💾 SAVE: save_models called 💾")
            
            # Ensure directory exists
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            
            # Save each sub-model
            for model_name, model in self.models.items():
                model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
                scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.joblib")
                
                joblib.dump(model, model_path)
                joblib.dump(self.scalers[model_name], scaler_path)
                print(f"💾 SAVE: Saved model {model_name}")
                logger.error(f"💾 SAVE: Saved model {model_name}")
            
            # Save meta-model
            if self.meta_model is not None:
                meta_model_path = os.path.join(self.model_dir, "meta_model.joblib")
                meta_scaler_path = os.path.join(self.model_dir, "meta_scaler.joblib")
                
                joblib.dump(self.meta_model, meta_model_path)
                joblib.dump(self.meta_scaler, meta_scaler_path)
                print("💾 SAVE: Saved meta-model")
                logger.error("💾 SAVE: Saved meta-model")
            
            # Save feature importance
            if self.feature_importance:
                importance_path = os.path.join(self.model_dir, "feature_importance.joblib")
                joblib.dump(self.feature_importance, importance_path)
                print("💾 SAVE: Saved feature importance")
                logger.error("💾 SAVE: Saved feature importance")
            
            # Save model metrics
            if self.model_metrics:
                metrics_path = os.path.join(self.model_dir, "model_metrics.joblib")
                joblib.dump(self.model_metrics, metrics_path)
                print("💾 SAVE: Saved model metrics")
                logger.error("💾 SAVE: Saved model metrics")
            
            # Save LSTM model if available with proper extension
            if TENSORFLOW_AVAILABLE and self.lstm_model is not None:
                # First try .keras extension (recommended for TF 2.x)
                try:
                    lstm_path = os.path.join(self.model_dir, "lstm_model.keras")
                    self.lstm_model.save(lstm_path, save_format='keras')
                    print(f"💾 SAVE: LSTM model saved to {lstm_path} in Keras format")
                    logger.error(f"💾 SAVE: LSTM model saved to {lstm_path} in Keras format")
                except Exception as e:
                    print(f"💾 SAVE: Error saving with .keras extension: {e}")
                    logger.error(f"💾 SAVE: Error saving with .keras extension: {e}")
                    try:
                        # Fall back to .h5 extension
                        lstm_path = os.path.join(self.model_dir, "lstm_model.h5") 
                        self.lstm_model.save(lstm_path, save_format='h5')
                        print(f"💾 SAVE: LSTM model saved to {lstm_path} in H5 format")
                        logger.error(f"💾 SAVE: LSTM model saved to {lstm_path} in H5 format")
                    except Exception as e2:
                        print(f"💾 SAVE: Error saving LSTM model with .h5 extension: {e2}")
                        logger.error(f"💾 SAVE: Error saving LSTM model with .h5 extension: {e2}")
            
            # Save LSTM scaler if available
            if self.lstm_scaler is not None:
                lstm_scaler_path = os.path.join(self.model_dir, "lstm_scaler.joblib")
                joblib.dump(self.lstm_scaler, lstm_scaler_path)
                print("💾 SAVE: LSTM scaler saved")
                logger.error("💾 SAVE: LSTM scaler saved")
            
            # Create and save model registry in JSON format for better compatibility
            registry = {
                'last_updated': datetime.now().isoformat(),
                'config': self.config,
                'all_feature_cols': self.all_feature_cols,
                'meta_feature_names': self.meta_feature_names,
                'lstm_feature_cols': self.lstm_feature_cols,
                'lstm_sequence_length': self.lstm_sequence_length,
                'models': {
                    model_name: {
                        'type': type(model).__name__,
                        'features': self.feature_subsets.get(model_name, [])
                    } for model_name, model in self.models.items()
                },
                'meta_model': {
                    'type': type(self.meta_model).__name__ if self.meta_model else None,
                }
            }
            
            # Save registry as JSON
            registry_path = os.path.join(self.model_dir, "model_registry.json")
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            
            print(f"💾 SAVE: Successfully saved all models to {self.model_dir}")
            logger.error(f"💾 SAVE: Successfully saved all models to {self.model_dir}")
            
            return True
            
        except Exception as e:
            print(f"💾 SAVE ERROR: Error saving models: {e}")
            print(f"💾 SAVE ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"💾 SAVE ERROR: Error saving models: {e}")
            logger.error(f"💾 SAVE ERROR TRACEBACK:\n{traceback.format_exc()}")
            return False
    
    def train_ensemble(self, data_frames, test_size=0.2, random_state=42, deep_learning=True):
        """
        Train the full ensemble of models including base models and meta-learner.
        
        Args:
            data_frames: List of pandas DataFrames with training data
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            deep_learning: Whether to include deep learning models
            
        Returns:
            dict: Dictionary with training metrics
        """
        print("🏋️ TRAIN: train_ensemble called 🏋️")
        logger.error("🏋️ TRAIN: train_ensemble called 🏋️")
        
        if not data_frames:
            print("🏋️ TRAIN: No data provided for training")
            logger.error("🏋️ TRAIN: No data provided for training")
            return {'success': False, 'message': 'No data provided'}
        
        try:
            # Update configuration
            self.config['test_size'] = test_size
            self.config['random_state'] = random_state
            
            # Combine all dataframes
            all_data = pd.concat(data_frames, ignore_index=True)
            print(f"🏋️ TRAIN: Combined {len(data_frames)} dataframes, total rows: {len(all_data)}")
            logger.error(f"🏋️ TRAIN: Combined {len(data_frames)} dataframes, total rows: {len(all_data)}")
            
            # Make sure we have the Label column
            if 'Label' not in all_data.columns:
                print("🏋️ TRAIN: Label column not found in training data")
                logger.error("🏋️ TRAIN: Label column not found in training data")
                return {'success': False, 'message': 'Label column not found'}
            
            # Get all feature columns (excluding non-features)
            exclude_cols = ['Label', 'Symbol', 'Timestamp', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            all_feature_cols = [col for col in all_data.columns if col not in exclude_cols]
            
            # Store feature columns for future reference
            self.all_feature_cols = all_feature_cols
            print(f"🏋️ TRAIN: Found {len(all_feature_cols)} feature columns")
            logger.error(f"🏋️ TRAIN: Found {len(all_feature_cols)} feature columns")
            
            if not all_feature_cols:
                print("🏋️ TRAIN: No feature columns found")
                logger.error("🏋️ TRAIN: No feature columns found")
                return {'success': False, 'message': 'No feature columns found'}
            
            # Sort by time first to avoid data leakage
            if 'Timestamp' in all_data.columns:
                all_data = all_data.sort_values('Timestamp')
                print("🏋️ TRAIN: Sorted data by Timestamp")
                logger.error("🏋️ TRAIN: Sorted data by Timestamp")
            elif 'Date' in all_data.columns:
                all_data = all_data.sort_values('Date')
                print("🏋️ TRAIN: Sorted data by Date")
                logger.error("🏋️ TRAIN: Sorted data by Date")
            
            # Use the last 20% as test set (time-aware split for financial data)
            train_idx = int(len(all_data) * (1 - test_size))
            train_data = all_data.iloc[:train_idx]
            test_data = all_data.iloc[train_idx:]
            
            print(f"🏋️ TRAIN: Train set: {len(train_data)} samples, Test set: {len(test_data)} samples")
            logger.error(f"🏋️ TRAIN: Train set: {len(train_data)} samples, Test set: {len(test_data)} samples")
            
            # Extract features and labels
            X_train = train_data[all_feature_cols]
            y_train = train_data['Label']
            
            X_test = test_data[all_feature_cols]
            y_test = test_data['Label']
            
            # Check for class imbalance and log it
            class_distribution = dict(train_data['Label'].value_counts())
            print(f"🏋️ TRAIN: Class distribution in training data: {class_distribution}")
            logger.error(f"🏋️ TRAIN: Class distribution in training data: {class_distribution}")
            
            # Detect and handle outliers/anomalies
            if self._detect_outliers(X_train):
                print("🏋️ TRAIN: Outliers detected in training data, using RobustScaler")
                logger.error("🏋️ TRAIN: Outliers detected in training data, using RobustScaler")
                scaler_type = "robust"
            else:
                scaler_type = "standard"
                print("🏋️ TRAIN: No significant outliers detected, using StandardScaler")
                logger.error("🏋️ TRAIN: No significant outliers detected, using StandardScaler")
            
            # Train multiple base models with different feature subsets
            print("🏋️ TRAIN: Training base models...")
            logger.error("🏋️ TRAIN: Training base models...")
            base_models_trained = self._train_base_models(X_train, y_train, X_test, y_test, all_feature_cols, scaler_type)
            
            if not base_models_trained:
                print("🏋️ TRAIN: Failed to train base models")
                logger.error("🏋️ TRAIN: Failed to train base models")
                return {'success': False, 'message': 'Base model training failed'}
            
            # Train LSTM model if enabled and TensorFlow is available
            lstm_trained = False
            if deep_learning and TENSORFLOW_AVAILABLE:
                try:
                    print("🏋️ TRAIN: Training LSTM model...")
                    logger.error("🏋️ TRAIN: Training LSTM model...")
                    lstm_trained = self._train_lstm_model(all_data, all_feature_cols)
                    print(f"🏋️ TRAIN: LSTM model training {'successful' if lstm_trained else 'failed'}")
                    logger.error(f"🏋️ TRAIN: LSTM model training {'successful' if lstm_trained else 'failed'}")
                except Exception as e:
                    print(f"🏋️ TRAIN: Error training LSTM model: {e}")
                    print(f"🏋️ TRAIN: Error traceback:\n{traceback.format_exc()}")
                    logger.error(f"🏋️ TRAIN: Error training LSTM model: {e}")
                    logger.error(f"🏋️ TRAIN: Error traceback:\n{traceback.format_exc()}")
                    lstm_trained = False
            
            # Create meta-features by getting predictions from base models
            print("🏋️ TRAIN: Creating meta-features...")
            logger.error("🏋️ TRAIN: Creating meta-features...")
            meta_features_train, meta_features_test = self._create_meta_features(X_train, X_test)
            
            # Store meta feature names for future reference
            if not meta_features_train.empty:
                self.meta_feature_names = meta_features_train.columns.tolist()
                print(f"🏋️ TRAIN: Meta-features: {self.meta_feature_names}")
                logger.error(f"🏋️ TRAIN: Meta-features: {self.meta_feature_names}")
            
            # Train meta-learner
            print("🏋️ TRAIN: Training meta-learner...")
            logger.error("🏋️ TRAIN: Training meta-learner...")
            meta_success = self._train_meta_learner(meta_features_train, y_train, meta_features_test, y_test)
            
            if not meta_success:
                print("🏋️ TRAIN: Meta-learner training failed, will use base models only")
                logger.error("🏋️ TRAIN: Meta-learner training failed, will use base models only")
            
            # Calculate ensemble metrics
            print("🏋️ TRAIN: Evaluating ensemble...")
            logger.error("🏋️ TRAIN: Evaluating ensemble...")
            ensemble_metrics = self._evaluate_ensemble(X_test, y_test)
            
            # Save all models
            print("🏋️ TRAIN: Saving models...")
            logger.error("🏋️ TRAIN: Saving models...")
            self.save_models()
            
            print(f"🏋️ TRAIN: Ensemble training completed with f2_score: {ensemble_metrics.get('f2_score', 0):.4f}")
            logger.error(f"🏋️ TRAIN: Ensemble training completed with f2_score: {ensemble_metrics.get('f2_score', 0):.4f}")
            
            return {
                'success': True,
                'message': 'Ensemble training completed successfully',
                'metrics': ensemble_metrics,
                'lstm_trained': lstm_trained,
                'feature_importance': self._get_top_features(10)
            }
            
        except Exception as e:
            print(f"🏋️ TRAIN ERROR: Error training ensemble: {e}")
            print(f"🏋️ TRAIN ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"🏋️ TRAIN ERROR: Error training ensemble: {e}")
            logger.error(f"🏋️ TRAIN ERROR TRACEBACK:\n{traceback.format_exc()}")
            return {'success': False, 'message': f'Error: {str(e)}'}
    
    def _detect_outliers(self, X):
        """
        Detect if there are significant outliers in the data.
        
        Args:
            X: Feature matrix
            
        Returns:
            bool: True if outliers are detected
        """
        try:
            print("🔍 OUTLIER: _detect_outliers called")
            logger.error("🔍 OUTLIER: _detect_outliers called")
            
            # Simple outlier detection by looking at skew
            numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
            if len(numerical_cols) == 0:
                print("🔍 OUTLIER: No numerical columns found")
                logger.error("🔍 OUTLIER: No numerical columns found")
                return False
                
            # Check skewness
            skew = X[numerical_cols].skew(numeric_only=True)
            high_skew = (abs(skew) > 3).sum()
            
            # If more than 20% of features have high skew, consider the data to have outliers
            result = high_skew > 0.2 * len(numerical_cols)
            print(f"🔍 OUTLIER: Detected {high_skew} columns with high skew out of {len(numerical_cols)}")
            logger.error(f"🔍 OUTLIER: Detected {high_skew} columns with high skew out of {len(numerical_cols)}")
            print(f"🔍 OUTLIER: Outliers detected: {result}")
            logger.error(f"🔍 OUTLIER: Outliers detected: {result}")
            return result
        except Exception as e:
            print(f"🔍 OUTLIER ERROR: {e}")
            logger.error(f"🔍 OUTLIER ERROR: {e}")
            return False
    
    def _train_base_models(self, X_train, y_train, X_test, y_test, all_feature_cols, scaler_type="standard"):
        """
        Train the base models with different feature subsets.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            all_feature_cols: All available feature columns
            scaler_type: Type of scaler to use ('standard' or 'robust')
            
        Returns:
            bool: Success or failure
        """
        try:
            print("🏋️‍♂️ BASE: _train_base_models called")
            logger.error("🏋️‍♂️ BASE: _train_base_models called")
            
            # Define feature subsets (domain knowledge based)
            feature_subsets = self._create_feature_subsets(all_feature_cols)
            print(f"🏋️‍♂️ BASE: Created {len(feature_subsets)} feature subsets")
            logger.error(f"🏋️‍♂️ BASE: Created {len(feature_subsets)} feature subsets")
            
            # Store feature subsets
            self.feature_subsets = feature_subsets
            
            # Define models to train for each feature subset
            model_configs = [
                # XGBoost models
                {'name': 'xgb_all', 'model': XGBClassifier(
                    use_label_encoder=False, 
                    eval_metric='logloss', 
                    random_state=self.config['random_state'],
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6
                ), 'subset': 'all_features'},
                
                # LightGBM models
                {'name': 'lgb_all', 'model': LGBMClassifier(
                    random_state=self.config['random_state'],
                    n_estimators=200,
                    learning_rate=0.05
                ), 'subset': 'all_features'},
                
                # Random Forest models
                {'name': 'rf_all', 'model': RandomForestClassifier(
                    n_estimators=200, 
                    max_depth=8,
                    random_state=self.config['random_state']
                ), 'subset': 'all_features'},
                
                # Gradient Boosting models
                {'name': 'gb_tech', 'model': GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=self.config['random_state']
                ), 'subset': 'technical'},
                
                # Specialized XGBoost models
                {'name': 'xgb_tech', 'model': XGBClassifier(
                    use_label_encoder=False, 
                    eval_metric='logloss', 
                    random_state=self.config['random_state']
                ), 'subset': 'technical'},
                
                {'name': 'xgb_vol', 'model': XGBClassifier(
                    use_label_encoder=False, 
                    eval_metric='logloss', 
                    random_state=self.config['random_state']
                ), 'subset': 'volatility'},
                
                # Specialized LightGBM models
                {'name': 'lgb_mom', 'model': LGBMClassifier(
                    random_state=self.config['random_state']
                ), 'subset': 'momentum'},
                
                {'name': 'lgb_price', 'model': LGBMClassifier(
                    random_state=self.config['random_state']
                ), 'subset': 'price_action'},
                
                # More Random Forest models
                {'name': 'rf_price', 'model': RandomForestClassifier(
                    n_estimators=150, 
                    random_state=self.config['random_state']
                ), 'subset': 'price_action'},
                
                # More XGBoost specialized models
                {'name': 'xgb_flow', 'model': XGBClassifier(
                    use_label_encoder=False, 
                    eval_metric='logloss', 
                    random_state=self.config['random_state']
                ), 'subset': 'order_flow'}
            ]
            
            # Ensure we only use feature subsets that exist
            model_configs = [cfg for cfg in model_configs if cfg['subset'] in feature_subsets and len(feature_subsets[cfg['subset']]) >= 5]
            print(f"🏋️‍♂️ BASE: Will train {len(model_configs)} base models")
            logger.error(f"🏋️‍♂️ BASE: Will train {len(model_configs)} base models")
            
            # Train each model
            for cfg in model_configs:
                subset_name = cfg['subset']
                model_name = cfg['name']
                model = cfg['model']
                
                # Get feature subset
                features = feature_subsets[subset_name]
                
                if not features:
                    print(f"🏋️‍♂️ BASE: No features for subset {subset_name}, skipping model {model_name}")
                    logger.error(f"🏋️‍♂️ BASE: No features for subset {subset_name}, skipping model {model_name}")
                    continue
                
                print(f"🏋️‍♂️ BASE: Training {model_name} with {len(features)} features from {subset_name}")
                logger.error(f"🏋️‍♂️ BASE: Training {model_name} with {len(features)} features from {subset_name}")
                
                # Extract feature subset
                X_train_subset = X_train[features]
                X_test_subset = X_test[features]
                
                # Create and fit scaler
                if scaler_type == "robust":
                    scaler = RobustScaler()
                else:
                    scaler = StandardScaler()
                
                X_train_scaled = scaler.fit_transform(X_train_subset)
                X_test_scaled = scaler.transform(X_test_subset)
                
                # Train model with early stopping if supported
                if hasattr(model, 'fit') and ('xgb' in model_name or 'lgb' in model_name):
                    # For XGBoost and LightGBM, use early stopping with a validation set
                    valid_size = 0.2
                    valid_samples = int(X_train_scaled.shape[0] * valid_size)
                    
                    # Split into train and validation
                    X_train_model = X_train_scaled[:-valid_samples]
                    X_valid = X_train_scaled[-valid_samples:]
                    y_train_model = y_train.iloc[:-valid_samples]
                    y_valid = y_train.iloc[-valid_samples:]
                    
                    # Create eval set
                    eval_set = [(X_valid, y_valid)]
                    
                    # Fit with early stopping
                    if 'xgb' in model_name:
                        model.fit(
                            X_train_model, y_train_model,
                            eval_set=eval_set,
                            early_stopping_rounds=20,
                            verbose=False
                        )
                    elif 'lgb' in model_name:
                        model.fit(
                            X_train_model, y_train_model,
                            eval_set=eval_set,
                            early_stopping_rounds=20,
                            verbose=False
                        )
                    else:
                        model.fit(X_train_scaled, y_train)
                else:
                    # Standard fit for other models
                    model.fit(X_train_scaled, y_train)
                
                # Store model and scaler
                self.models[model_name] = model
                self.scalers[model_name] = scaler
                self.feature_subsets[model_name] = features
                print(f"🏋️‍♂️ BASE: Model {model_name} trained successfully")
                logger.error(f"🏋️‍♂️ BASE: Model {model_name} trained successfully")
                
                # Calculate and store metrics
                y_pred = model.predict(X_test_scaled)
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'f2_score': fbeta_score(y_test, y_pred, beta=2),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                }
                
                # Add AUC if model supports predict_proba
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(X_test_scaled)[:, 1]
                        metrics['auc'] = roc_auc_score(y_test, proba)
                    except:
                        metrics['auc'] = 0.5
                else:
                    metrics['auc'] = 0.5
                
                self.model_metrics[model_name] = metrics
                
                print(f"🏋️‍♂️ BASE: Model {model_name} - f2_score: {metrics['f2_score']:.4f}, AUC: {metrics['auc']:.4f}")
                logger.error(f"🏋️‍♂️ BASE: Model {model_name} - f2_score: {metrics['f2_score']:.4f}, AUC: {metrics['auc']:.4f}")
                
                # Store feature importances if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(zip(features, model.feature_importances_))
                    print(f"🏋️‍♂️ BASE: Stored feature importances for {model_name}")
                    logger.error(f"🏋️‍♂️ BASE: Stored feature importances for {model_name}")
            
            print("🏋️‍♂️ BASE: All base models trained successfully")
            logger.error("🏋️‍♂️ BASE: All base models trained successfully")
            return True
            
        except Exception as e:
            print(f"🏋️‍♂️ BASE ERROR: Error training base models: {e}")
            print(f"🏋️‍♂️ BASE ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"🏋️‍♂️ BASE ERROR: Error training base models: {e}")
            logger.error(f"🏋️‍♂️ BASE ERROR TRACEBACK:\n{traceback.format_exc()}")
            return False
    
    def _create_feature_subsets(self, all_feature_cols):
        """
        Create feature subsets from all feature columns.
        
        Args:
            all_feature_cols: All available feature columns
            
        Returns:
            dict: Dictionary of feature subsets
        """
        print("📊 SUBSET: _create_feature_subsets called")
        logger.error("📊 SUBSET: _create_feature_subsets called")
        
        # 1. Technical indicators subset
        tech_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
            'sma', 'ema', 'macd', 'rsi', 'bband', 'stoch', 'adx', 'atr', 'obv', 'cci', 'willr'
        ])]
        
        # 2. Volatility indicators subset
        vol_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
            'volatility', 'garch', 'atr', 'range', 'std', 'bband', 'vix', 'variance'
        ])]
        
        # 3. Momentum indicators subset
        mom_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
            'momentum', 'rsi', 'macd', 'roc', 'willr', 'stoch', 'trix', 'ao', 'acceleration'
        ])]
        
        # 4. Price action subset
        price_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
            'close', 'price', 'open', 'high', 'low', 'range', 'lag', 'change', 'return', 'gap'
        ])]
        
        # 5. Volume indicators subset
        vol_ind_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
            'volume', 'vwap', 'obv', 'mfi', 'ad', 'cmf', 'vpt', 'pvt'
        ])]
        
        # 6. Sentiment and options subset
        sent_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
            'sentiment', 'social', 'news', 'option', 'call', 'put', 'unusual', 'headline', 'fear', 'greed'
        ])]
        
        # 7. Market regime subset
        regime_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
            'regime', 'trend', 'bull', 'bear', 'sideways', 'correction', 'recovery'
        ])]
        
        # 8. Order flow subset
        order_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
            'flow', 'imbalance', 'delta', 'vwap', 'dark', 'pressure', 'market', 'limit', 'bid', 'ask', 'spread'
        ])]
        
        # 9. Mean reversion subset
        mean_rev_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
            'zscore', 'mean', 'reversion', 'overextended', 'overbought', 'oversold'
        ])]
        
        # 10. Correlation and relative strength subset
        corr_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
            'correlation', 'relative', 'strength', 'beta', 'alpha', 'sector', 'industry'
        ])]
        
        # Combine into feature subsets
        feature_subsets = {
            'all_features': all_feature_cols,
            'technical': tech_features,
            'volatility': vol_features,
            'momentum': mom_features,
            'price_action': price_features,
            'volume': vol_ind_features,
            'sentiment': sent_features,
            'market_regime': regime_features,
            'order_flow': order_features,
            'mean_reversion': mean_rev_features,
            'correlation': corr_features
        }
        
        # Filter out empty subsets and ensure minimum features
        feature_subsets = {k: v for k, v in feature_subsets.items() if len(v) >= 5}
        
        # Log feature subset counts
        for name, features in feature_subsets.items():
            print(f"📊 SUBSET: {name} has {len(features)} features")
            logger.error(f"📊 SUBSET: {name} has {len(features)} features")
        
        # If we don't have enough specialized features, use random feature subsets
        if len(feature_subsets) < 3:
            print("📊 SUBSET: Not enough specialized feature subsets, using random subsets")
            logger.error("📊 SUBSET: Not enough specialized feature subsets, using random subsets")
            np.random.seed(self.config['random_state'])  # For reproducibility
            n_features = min(50, len(all_feature_cols))
            feature_subsets = {
                'all_features': all_feature_cols,
                'subset1': np.random.choice(all_feature_cols, n_features, replace=False).tolist(),
                'subset2': np.random.choice(all_feature_cols, n_features, replace=False).tolist()
            }
            
        return feature_subsets
    
    def _create_meta_features(self, X_train, X_test):
        """
        Create meta-features by getting predictions from all base models.
        These meta-features will be used to train the meta-learner.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            tuple: (meta_features_train, meta_features_test)
        """
        try:
            print("🧩 META: _create_meta_features called")
            logger.error("🧩 META: _create_meta_features called")
            
            meta_features_train = pd.DataFrame()
            meta_features_test = pd.DataFrame()
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                print(f"🧩 META: Getting predictions from {model_name}...")
                logger.error(f"🧩 META: Getting predictions from {model_name}...")
                
                # Get feature subset and scaler for this model
                features = self.feature_subsets[model_name]
                scaler = self.scalers[model_name]
                
                # Scale features
                X_train_scaled = scaler.transform(X_train[features])
                X_test_scaled = scaler.transform(X_test[features])
                
                # Get probability predictions if available
                if hasattr(model, 'predict_proba'):
                    train_preds = model.predict_proba(X_train_scaled)[:, 1]
                    test_preds = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    train_preds = model.predict(X_train_scaled)
                    test_preds = model.predict(X_test_scaled)
                
                # Add predictions as meta-features
                meta_features_train[f"{model_name}_prob"] = train_preds
                meta_features_test[f"{model_name}_prob"] = test_preds
                print(f"🧩 META: Added {model_name} predictions to meta-features")
                logger.error(f"🧩 META: Added {model_name} predictions to meta-features")
            
            # Add LSTM predictions if available
            if TENSORFLOW_AVAILABLE and self.lstm_model is not None and self.lstm_scaler is not None:
                try:
                    print("🧩 META: Getting LSTM predictions...")
                    logger.error("🧩 META: Getting LSTM predictions...")
                    # Prepare LSTM input for all samples
                    lstm_X_train = self._prepare_lstm_input(X_train)
                    lstm_X_test = self._prepare_lstm_input(X_test)
                    
                    if lstm_X_train is not None and lstm_X_test is not None:
                        # Get LSTM predictions
                        train_preds = self.lstm_model.predict(lstm_X_train).flatten()
                        test_preds = self.lstm_model.predict(lstm_X_test).flatten()
                        
                        # Validate predictions have the right shape
                        if len(train_preds) == len(X_train) and len(test_preds) == len(X_test):
                            # Add as meta-features
                            meta_features_train['lstm_prob'] = train_preds
                            meta_features_test['lstm_prob'] = test_preds
                            print("🧩 META: Added LSTM predictions to meta-features")
                            logger.error("🧩 META: Added LSTM predictions to meta-features")
                        else:
                            print(f"🧩 META: LSTM predictions shape mismatch: train={len(train_preds)}, expected={len(X_train)}")
                            logger.error(f"🧩 META: LSTM predictions shape mismatch: train={len(train_preds)}, expected={len(X_train)}")
                            # Still add lstm_prob with default values
                            meta_features_train['lstm_prob'] = 0.5
                            meta_features_test['lstm_prob'] = 0.5
                            print("🧩 META: Added default LSTM values (0.5) to meta-features")
                            logger.error("🧩 META: Added default LSTM values (0.5) to meta-features")
                    else:
                        # Add lstm_prob with default values
                        meta_features_train['lstm_prob'] = 0.5
                        meta_features_test['lstm_prob'] = 0.5
                        print("🧩 META: Added default LSTM values (0.5) to meta-features")
                        logger.error("🧩 META: Added default LSTM values (0.5) to meta-features")
                except Exception as e:
                    print(f"🧩 META: Error getting LSTM predictions: {e}")
                    print(f"🧩 META: Error traceback:\n{traceback.format_exc()}")
                    logger.error(f"🧩 META: Error getting LSTM predictions: {e}")
                    logger.error(f"🧩 META: Error traceback:\n{traceback.format_exc()}")
                    # Still add lstm_prob with default values
                    meta_features_train['lstm_prob'] = 0.5
                    meta_features_test['lstm_prob'] = 0.5
                    print("🧩 META: Added default LSTM values (0.5) to meta-features due to error")
                    logger.error("🧩 META: Added default LSTM values (0.5) to meta-features due to error")
            else:
                # Always add lstm_prob even if LSTM is not available
                meta_features_train['lstm_prob'] = 0.5
                meta_features_test['lstm_prob'] = 0.5
                print("🧩 META: Added default LSTM predictions (0.5) to meta-features")
                logger.error("🧩 META: Added default LSTM predictions (0.5) to meta-features")
            
            print(f"🧩 META: Created meta-features with shape: {meta_features_train.shape}")
            logger.error(f"🧩 META: Created meta-features with shape: {meta_features_train.shape}")
            
            # Debug log the metadata features columns
            print(f"🧩 META: Meta-features train columns: {meta_features_train.columns.tolist()}")
            logger.error(f"🧩 META: Meta-features train columns: {meta_features_train.columns.tolist()}")
            print(f"🧩 META: Meta-features test columns: {meta_features_test.columns.tolist()}")
            logger.error(f"🧩 META: Meta-features test columns: {meta_features_test.columns.tolist()}")
            
            return meta_features_train, meta_features_test
            
        except Exception as e:
            print(f"🧩 META ERROR: Error creating meta-features: {e}")
            print(f"🧩 META ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"🧩 META ERROR: Error creating meta-features: {e}")
            logger.error(f"🧩 META ERROR TRACEBACK:\n{traceback.format_exc()}")
            # Return empty dataframes if there's an error
            return pd.DataFrame(), pd.DataFrame()

    def _train_meta_learner(self, meta_features_train, y_train, meta_features_test, y_test):
        """
        Train the meta-learner model using meta-features.
        
        Args:
            meta_features_train: Meta-features for training
            y_train: Training labels
            meta_features_test: Meta-features for testing
            y_test: Test labels
            
        Returns:
            bool: Success or failure
        """
        try:
            print("🤖 META-LEARN: _train_meta_learner called")
            logger.error("🤖 META-LEARN: _train_meta_learner called")
            
            if meta_features_train.empty or meta_features_test.empty:
                print("🤖 META-LEARN: Empty meta-features, cannot train meta-learner")
                logger.error("🤖 META-LEARN: Empty meta-features, cannot train meta-learner")
                return False
                
            # First, ensure meta-features have consistent columns across train and test
            all_columns = set(meta_features_train.columns) | set(meta_features_test.columns)
            
            # Add missing columns with zeros
            for col in all_columns:
                if col not in meta_features_train.columns:
                    meta_features_train[col] = 0.5  # Default value
                    print(f"🤖 META-LEARN: Added missing column {col} to train set")
                    logger.error(f"🤖 META-LEARN: Added missing column {col} to train set")
                if col not in meta_features_test.columns:
                    meta_features_test[col] = 0.5  # Default value
                    print(f"🤖 META-LEARN: Added missing column {col} to test set")
                    logger.error(f"🤖 META-LEARN: Added missing column {col} to test set")
            
            # Create and fit scaler for meta-features
            self.meta_scaler = StandardScaler()
            meta_train_scaled = self.meta_scaler.fit_transform(meta_features_train)
            meta_test_scaled = self.meta_scaler.transform(meta_features_test)
            print("🤖 META-LEARN: Applied StandardScaler to meta-features")
            logger.error("🤖 META-LEARN: Applied StandardScaler to meta-features")
            
            # Verify we have 'lstm_prob' in our features
            if 'lstm_prob' in meta_features_train.columns:
                print("🤖 META-LEARN: 'lstm_prob' is present in meta-features")
                logger.error("🤖 META-LEARN: 'lstm_prob' is present in meta-features")
            else:
                print("🤖 META-LEARN: WARNING - 'lstm_prob' is MISSING from meta-features!")
                logger.error("🤖 META-LEARN: WARNING - 'lstm_prob' is MISSING from meta-features!")
            
            # Select meta-learner based on configuration
            meta_learner_type = self.config.get('meta_learner_type', 'xgboost')
            print(f"🤖 META-LEARN: Using {meta_learner_type} for meta-learner")
            logger.error(f"🤖 META-LEARN: Using {meta_learner_type} for meta-learner")
            
            if meta_learner_type == 'xgboost':
                # XGBoost meta-learner
                self.meta_model = XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=self.config['random_state']
                )
            elif meta_learner_type == 'lightgbm':
                # LightGBM meta-learner
                self.meta_model = LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=32,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.config['random_state']
                )
            elif meta_learner_type == 'randomforest':
                # Random Forest meta-learner
                self.meta_model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.config['random_state']
                )
            elif meta_learner_type == 'voting':
                # Voting meta-learner (combines multiple meta-models)
                meta_models = [
                    ('xgb', XGBClassifier(
                        n_estimators=100, learning_rate=0.05, max_depth=3,
                        use_label_encoder=False, eval_metric='logloss',
                        random_state=self.config['random_state']
                    )),
                    ('lgb', LGBMClassifier(
                        n_estimators=100, learning_rate=0.05, num_leaves=32,
                        random_state=self.config['random_state']
                    )),
                    ('rf', RandomForestClassifier(
                        n_estimators=100, max_depth=5,
                        random_state=self.config['random_state']
                    ))
                ]
                
                self.meta_model = VotingClassifier(
                    estimators=meta_models,
                    voting='soft',
                    weights=[2, 1, 1]
                )
            else:
                # Default to XGBoost
                self.meta_model = XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=self.config['random_state']
                )
            
            # Train meta-model with early stopping if supported
            if hasattr(self.meta_model, 'fit') and hasattr(self.meta_model, 'predict_proba'):
                # For models supporting early stopping
                if meta_learner_type in ['xgboost', 'lightgbm']:
                    # Split for validation
                    valid_size = 0.2
                    valid_samples = int(meta_train_scaled.shape[0] * valid_size)
                    
                    X_train_meta = meta_train_scaled[:-valid_samples]
                    X_valid_meta = meta_train_scaled[-valid_samples:]
                    y_train_meta = y_train.iloc[:-valid_samples]
                    y_valid_meta = y_train.iloc[-valid_samples:]
                    
                    eval_set = [(X_valid_meta, y_valid_meta)]
                    
                    print(f"🤖 META-LEARN: Training {meta_learner_type} with early stopping")
                    logger.error(f"🤖 META-LEARN: Training {meta_learner_type} with early stopping")
                    if meta_learner_type == 'xgboost':
                        self.meta_model.fit(
                            X_train_meta, y_train_meta,
                            eval_set=eval_set,
                            early_stopping_rounds=20,
                            verbose=False
                        )
                    elif meta_learner_type == 'lightgbm':
                        self.meta_model.fit(
                            X_train_meta, y_train_meta,
                            eval_set=eval_set,
                            early_stopping_rounds=20,
                            verbose=False
                        )
                else:
                    # Standard fit for other models
                    print(f"🤖 META-LEARN: Training {meta_learner_type} with standard fit")
                    logger.error(f"🤖 META-LEARN: Training {meta_learner_type} with standard fit")
                    self.meta_model.fit(meta_train_scaled, y_train)
            else:
                # Standard fit
                print("🤖 META-LEARN: Training meta-model with standard fit")
                logger.error("🤖 META-LEARN: Training meta-model with standard fit")
                self.meta_model.fit(meta_train_scaled, y_train)
            
            # Store feature names from training
            self.meta_feature_names = meta_features_train.columns.tolist()
            print(f"🤖 META-LEARN: Stored meta_feature_names: {self.meta_feature_names}")
            logger.error(f"🤖 META-LEARN: Stored meta_feature_names: {self.meta_feature_names}")
            
            # If the model has feature_names_in_, try to check if it matches our stored feature names
            if hasattr(self.meta_model, 'feature_names_in_'):
                print(f"🤖 META-LEARN: Model's feature_names_in_: {self.meta_model.feature_names_in_}")
                logger.error(f"🤖 META-LEARN: Model's feature_names_in_: {self.meta_model.feature_names_in_}")
                
                # Check for mismatch
                if len(self.meta_feature_names) != len(self.meta_model.feature_names_in_):
                    print("🤖 META-LEARN: WARNING - Length mismatch in feature names!")
                    logger.error("🤖 META-LEARN: WARNING - Length mismatch in feature names!")
            
            # Evaluate meta-model
            y_pred = self.meta_model.predict(meta_test_scaled)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'f2_score': fbeta_score(y_test, y_pred, beta=2),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
            }
            
            # Add AUC if available
            if hasattr(self.meta_model, 'predict_proba'):
                try:
                    proba = self.meta_model.predict_proba(meta_test_scaled)[:, 1]
                    metrics['auc'] = roc_auc_score(y_test, proba)
                except:
                    metrics['auc'] = 0.5
            else:
                metrics['auc'] = 0.5
            
            # Add confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            self.model_metrics['meta_learner'] = metrics
            
            print(f"🤖 META-LEARN: Meta-learner trained with f2_score: {metrics['f2_score']:.4f}, AUC: {metrics.get('auc', 0):.4f}")
            logger.error(f"🤖 META-LEARN: Meta-learner trained with f2_score: {metrics['f2_score']:.4f}, AUC: {metrics.get('auc', 0):.4f}")
            
            return True
            
        except Exception as e:
            print(f"🤖 META-LEARN ERROR: Error training meta-learner: {e}")
            print(f"🤖 META-LEARN ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"🤖 META-LEARN ERROR: Error training meta-learner: {e}")
            logger.error(f"🤖 META-LEARN ERROR TRACEBACK:\n{traceback.format_exc()}")
            return False

    def _evaluate_ensemble(self, X_test, y_test):
        """
        Evaluate the complete ensemble on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            print("📊 EVAL: _evaluate_ensemble called")
            logger.error("📊 EVAL: _evaluate_ensemble called")
            
            # Initialize arrays for predictions
            y_preds = []
            probas = []
            
            # Get ensemble predictions for each sample
            for i in range(len(X_test)):
                # Get prediction for this sample
                sample = X_test.iloc[[i]] if hasattr(X_test, 'iloc') else X_test[i:i+1]
                prob, pred = self.predict_with_ensemble(sample)
                
                y_preds.append(pred)
                probas.append(prob)
                
                if i % 10 == 0:
                    print(f"📊 EVAL: Evaluated {i}/{len(X_test)} samples")
                    logger.error(f"📊 EVAL: Evaluated {i}/{len(X_test)} samples")
            
            # Convert to numpy arrays
            y_preds = np.array(y_preds)
            probas = np.array(probas)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_preds),
                'f1_score': f1_score(y_test, y_preds),
                'f2_score': fbeta_score(y_test, y_preds, beta=2),
                'precision': precision_score(y_test, y_preds),
                'recall': recall_score(y_test, y_preds),
                'auc': roc_auc_score(y_test, probas)
            }
            
            # Add confusion matrix
            cm = confusion_matrix(y_test, y_preds)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Store in model_metrics
            self.model_metrics['ensemble'] = metrics
            
            print(f"📊 EVAL: Ensemble evaluation: {metrics}")
            logger.error(f"📊 EVAL: Ensemble evaluation: {metrics}")
            
            return metrics
        
        except Exception as e:
            print(f"📊 EVAL ERROR: Error evaluating ensemble: {e}")
            print(f"📊 EVAL ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"📊 EVAL ERROR: Error evaluating ensemble: {e}")
            logger.error(f"📊 EVAL ERROR TRACEBACK:\n{traceback.format_exc()}")
            return {'error': str(e)}

    def _build_lstm_model(self, input_shape, hyperparams=None):
        """
        Build an advanced LSTM model architecture with attention.
        
        Args:
            input_shape: Input shape for the model (sequence_length, n_features)
            hyperparams: Optional hyperparameters for the model
            
        Returns:
            keras.Model: Compiled LSTM model
        """
        if not TENSORFLOW_AVAILABLE:
            print("🧠 LSTM BUILD: TensorFlow not available, cannot build LSTM model")
            logger.error("🧠 LSTM BUILD: TensorFlow not available, cannot build LSTM model")
            return None
            
        try:
            print("🧠 LSTM BUILD: Building LSTM model...")
            logger.error("🧠 LSTM BUILD: Building LSTM model...")
            
            # Default hyperparameters
            hp = {
                'lstm_units1': 64,
                'lstm_units2': 32,
                'dense_units': 16,
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'use_attention': self.config.get('use_attention', True),
                'use_bidirectional': True
            }
            
            # Update with provided hyperparameters if any
            if hyperparams:
                hp.update(hyperparams)
            
            print(f"🧠 LSTM BUILD: Hyperparameters: {hp}")
            logger.error(f"🧠 LSTM BUILD: Hyperparameters: {hp}")
            
            # Create model
            model = Sequential()
            
            # First LSTM layer
            if hp['use_bidirectional']:
                # Bidirectional LSTM with return sequences for attention
                model.add(Bidirectional(
                    LSTM(units=hp['lstm_units1'], 
                         return_sequences=True),
                    input_shape=input_shape
                ))
                print("🧠 LSTM BUILD: Added Bidirectional LSTM layer")
                logger.error("🧠 LSTM BUILD: Added Bidirectional LSTM layer")
            else:
                # Standard LSTM
                model.add(LSTM(
                    units=hp['lstm_units1'],
                    return_sequences=True,
                    input_shape=input_shape
                ))
                print("🧠 LSTM BUILD: Added standard LSTM layer")
                logger.error("🧠 LSTM BUILD: Added standard LSTM layer")
            
            # Add normalization and dropout
            model.add(LayerNormalization())  # Layer normalization for better stability
            model.add(Dropout(hp['dropout_rate']))
            print("🧠 LSTM BUILD: Added normalization and dropout")
            logger.error("🧠 LSTM BUILD: Added normalization and dropout")
            
            # Add attention mechanism if requested
            if hp['use_attention']:
                print("🧠 LSTM BUILD: Adding attention mechanism")
                logger.error("🧠 LSTM BUILD: Adding attention mechanism")
                # Implement a simple self-attention mechanism
                # Replace sequential model with functional API for attention
                inputs = keras.Input(shape=input_shape)
                x = Bidirectional(LSTM(units=hp['lstm_units1'], return_sequences=True))(inputs)
                x = LayerNormalization()(x)
                x = Dropout(hp['dropout_rate'])(x)
                
                # Apply attention
                attention = Conv1D(1, kernel_size=1)(x)
                attention = keras.activations.tanh(attention)
                attention_weights = keras.layers.Softmax(axis=1)(attention)
                context = keras.layers.Multiply()([x, attention_weights])
                context = keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
                
                # Continue with dense layers
                x = Dense(hp['dense_units'], activation='relu')(context)
                x = Dropout(hp['dropout_rate'])(x)
                outputs = Dense(1, activation='sigmoid')(x)
                
                # Create model
                model = keras.Model(inputs=inputs, outputs=outputs)
                print("🧠 LSTM BUILD: Built LSTM model with attention")
                logger.error("🧠 LSTM BUILD: Built LSTM model with attention")
            else:
                # Standard setup without attention
                model.add(LSTM(units=hp['lstm_units2'], return_sequences=False))
                model.add(BatchNormalization())
                model.add(Dropout(hp['dropout_rate']))
                model.add(Dense(hp['dense_units'], activation='relu'))
                model.add(Dense(1, activation='sigmoid'))
                print("🧠 LSTM BUILD: Built standard LSTM model")
                logger.error("🧠 LSTM BUILD: Built standard LSTM model")
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=hp['learning_rate']),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            print("🧠 LSTM BUILD: LSTM model compiled successfully")
            logger.error("🧠 LSTM BUILD: LSTM model compiled successfully")
            return model
            
        except Exception as e:
            print(f"🧠 LSTM BUILD ERROR: Error building LSTM model: {e}")
            print(f"🧠 LSTM BUILD ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"🧠 LSTM BUILD ERROR: Error building LSTM model: {e}")
            logger.error(f"🧠 LSTM BUILD ERROR TRACEBACK:\n{traceback.format_exc()}")
            return None
    def _train_lstm_model(self, all_data, all_feature_cols, sequence_length=None):
        """
        Train an LSTM model for sequence prediction.
        
        Args:
            all_data: Full dataset
            all_feature_cols: All available feature columns
            sequence_length: Length of sequences to use (default: use self.lstm_sequence_length)
            
        Returns:
            bool: Success or failure
        """
        if not TENSORFLOW_AVAILABLE:
            print("🧠 LSTM TRAIN: TensorFlow not available, skipping LSTM training")
            logger.error("🧠 LSTM TRAIN: TensorFlow not available, skipping LSTM training")
            return False
        
        try:
            print("🧠 LSTM TRAIN: Training LSTM model...")
            logger.error("🧠 LSTM TRAIN: Training LSTM model...")
            
            # Set sequence length if provided
            if sequence_length is not None:
                self.lstm_sequence_length = sequence_length
                print(f"🧠 LSTM TRAIN: Using sequence length: {self.lstm_sequence_length}")
                logger.error(f"🧠 LSTM TRAIN: Using sequence length: {self.lstm_sequence_length}")
            
            # In production, we'd use more robust feature selection for LSTM
            # Here we're simplifying by using a subset of important features
            if self.feature_importance and self.config.get('use_feature_selection', True):
                # Get top 20 features by importance if available
                top_features = self._get_top_features(20)
                if top_features:
                    lstm_feature_cols = [f for f in top_features if f in all_feature_cols]
                    if len(lstm_feature_cols) >= 5:
                        print(f"🧠 LSTM TRAIN: Using {len(lstm_feature_cols)} top features for LSTM")
                        logger.error(f"🧠 LSTM TRAIN: Using {len(lstm_feature_cols)} top features for LSTM")
                    else:
                        lstm_feature_cols = all_feature_cols
                        print(f"🧠 LSTM TRAIN: Not enough top features, using all {len(all_feature_cols)} features")
                        logger.error(f"🧠 LSTM TRAIN: Not enough top features, using all {len(all_feature_cols)} features")
                else:
                    lstm_feature_cols = all_feature_cols
                    print(f"🧠 LSTM TRAIN: No feature importance available, using all {len(all_feature_cols)} features")
                    logger.error(f"🧠 LSTM TRAIN: No feature importance available, using all {len(all_feature_cols)} features")
            else:
                lstm_feature_cols = all_feature_cols
                print(f"🧠 LSTM TRAIN: Using all {len(all_feature_cols)} features")
                logger.error(f"🧠 LSTM TRAIN: Using all {len(all_feature_cols)} features")
            
            # Store LSTM feature columns
            self.lstm_feature_cols = lstm_feature_cols
            
            # Prepare data for LSTM
            print("🧠 LSTM TRAIN: Preparing data for LSTM...")
            logger.error("🧠 LSTM TRAIN: Preparing data for LSTM...")
            data_subset = all_data[self.lstm_feature_cols + ['Label']].copy()
            
            # Handle missing values if any
            data_subset = data_subset.fillna(method='ffill')
            data_subset = data_subset.fillna(method='bfill')
            data_subset = data_subset.fillna(0)  # Fill any remaining NaNs with 0
            
            # Create a robust scaler for LSTM features
            self.lstm_scaler = StandardScaler()
            data_subset[self.lstm_feature_cols] = self.lstm_scaler.fit_transform(data_subset[self.lstm_feature_cols])
            print("🧠 LSTM TRAIN: Scaled features with StandardScaler")
            logger.error("🧠 LSTM TRAIN: Scaled features with StandardScaler")
            
            # Create sequences
            print("🧠 LSTM TRAIN: Creating input sequences...")
            logger.error("🧠 LSTM TRAIN: Creating input sequences...")
            X_sequences, y = self._prepare_sequence_data(data_subset, self.lstm_feature_cols)
            
            if X_sequences is None or y is None or len(X_sequences) == 0:
                print("🧠 LSTM TRAIN: Could not prepare sequence data for LSTM")
                logger.error("🧠 LSTM TRAIN: Could not prepare sequence data for LSTM")
                return False
                
            # Split into train, validation, and test
            train_size = int(len(X_sequences) * 0.7)
            val_size = int(len(X_sequences) * 0.15)
            
            X_train = X_sequences[:train_size]
            y_train = y[:train_size]
            
            X_val = X_sequences[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            
            X_test = X_sequences[train_size + val_size:]
            y_test = y[train_size + val_size:]
            
            print(f"🧠 LSTM TRAIN: Split data - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            logger.error(f"🧠 LSTM TRAIN: Split data - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Build and compile LSTM model
            print("🧠 LSTM TRAIN: Building and compiling LSTM model...")
            logger.error("🧠 LSTM TRAIN: Building and compiling LSTM model...")
            input_shape = (self.lstm_sequence_length, X_train.shape[2])
            self.lstm_model = self._build_lstm_model(input_shape)
            
            if self.lstm_model is None:
                print("🧠 LSTM TRAIN: Failed to build LSTM model")
                logger.error("🧠 LSTM TRAIN: Failed to build LSTM model")
                return False
            
            # Create callbacks for LSTM training
            print("🧠 LSTM TRAIN: Setting up training callbacks...")
            logger.error("🧠 LSTM TRAIN: Setting up training callbacks...")
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.get('early_stopping_patience', 10),
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001
                )
            ]
            
            # Add model checkpoint if directory exists
            checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                
            checkpoint_path = os.path.join(checkpoint_dir, 'lstm_checkpoint.keras')
            callbacks.append(
                ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True
                )
            )
            
            # Train model
            print("🧠 LSTM TRAIN: Starting LSTM training...")
            logger.error("🧠 LSTM TRAIN: Starting LSTM training...")
            history = self.lstm_model.fit(
                X_train, y_train,
                epochs=self.config.get('lstm_epochs', 50),
                batch_size=self.config.get('lstm_batch_size', 32),
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Record trained epochs
            self.model_epochs['lstm'] = len(history.history['loss'])
            print(f"🧠 LSTM TRAIN: LSTM model trained for {self.model_epochs['lstm']} epochs")
            logger.error(f"🧠 LSTM TRAIN: LSTM model trained for {self.model_epochs['lstm']} epochs")
            
            # Evaluate model
            print("🧠 LSTM TRAIN: Evaluating LSTM model...")
            logger.error("🧠 LSTM TRAIN: Evaluating LSTM model...")
            results = self.lstm_model.evaluate(X_test, y_test)
            
            # Make predictions for test set
            y_pred_proba = self.lstm_model.predict(X_test).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate detailed metrics
            metrics = {
                'loss': float(results[0]),
                'accuracy': float(results[1]),
                'f1_score': float(f1_score(y_test, y_pred)),
                'f2_score': float(fbeta_score(y_test, y_pred, beta=2)),
                'precision': float(precision_score(y_test, y_pred)),
                'recall': float(recall_score(y_test, y_pred)),
                'auc': float(roc_auc_score(y_test, y_pred_proba))
            }
            
            # Add confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Store metrics
            self.model_metrics['lstm'] = metrics
            
            print(f"🧠 LSTM TRAIN: LSTM model trained with accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
            logger.error(f"🧠 LSTM TRAIN: LSTM model trained with accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"🧠 LSTM TRAIN ERROR: Error training LSTM model: {e}")
            print(f"🧠 LSTM TRAIN ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"🧠 LSTM TRAIN ERROR: Error training LSTM model: {e}")
            logger.error(f"🧠 LSTM TRAIN ERROR TRACEBACK:\n{traceback.format_exc()}")
            return False
    
    def _prepare_sequence_data(self, df, feature_cols, target_col='Label', drop_y_na=True):
        """
        Prepare sequence data for LSTM model.
        
        Args:
            df: DataFrame with features and target
            feature_cols: Feature columns to use
            target_col: Target column name
            drop_y_na: Whether to drop rows with NA in target
            
        Returns:
            tuple: (X_sequences, y_target) or (None, None) if error
        """
        try:
            print("🔄 SEQ: _prepare_sequence_data called")
            logger.error("🔄 SEQ: _prepare_sequence_data called")
            
            # Check if DataFrame is not empty
            if df.empty:
                print("🔄 SEQ: Empty DataFrame provided for sequence preparation")
                logger.error("🔄 SEQ: Empty DataFrame provided for sequence preparation")
                return None, None
                
            # Must be sorted by time (assume it's already sorted)
            if 'Timestamp' in df.columns:
                df = df.sort_values('Timestamp').reset_index(drop=True)
                print("🔄 SEQ: Sorted data by Timestamp")
                logger.error("🔄 SEQ: Sorted data by Timestamp")
            elif 'Date' in df.columns:
                df = df.sort_values('Date').reset_index(drop=True)
                print("🔄 SEQ: Sorted data by Date")
                logger.error("🔄 SEQ: Sorted data by Date")
            
            # Handle missing values in target if needed
            if drop_y_na and df[target_col].isna().any():
                df = df.dropna(subset=[target_col])
                print(f"🔄 SEQ: Dropped rows with NaN in {target_col}")
                logger.error(f"🔄 SEQ: Dropped rows with NaN in {target_col}")
            
            # Get features and target
            X = df[feature_cols].values
            y = df[target_col].values
            
            # Create sequences
            X_sequences = []
            y_target = []
            
            # Check if we have enough data for at least one sequence
            if len(X) <= self.lstm_sequence_length:
                print(f"🔄 SEQ: Not enough data points ({len(X)}) for sequence length {self.lstm_sequence_length}")
                logger.error(f"🔄 SEQ: Not enough data points ({len(X)}) for sequence length {self.lstm_sequence_length}")
                return None, None
            
            for i in range(len(X) - self.lstm_sequence_length):
                X_sequences.append(X[i:(i + self.lstm_sequence_length)])
                y_target.append(y[i + self.lstm_sequence_length])
            
            # Convert to numpy arrays
            X_sequences = np.array(X_sequences)
            y_target = np.array(y_target)
            
            print(f"🔄 SEQ: Created {len(X_sequences)} sequences with shape {X_sequences.shape}")
            logger.error(f"🔄 SEQ: Created {len(X_sequences)} sequences with shape {X_sequences.shape}")
            
            return X_sequences, y_target
            
        except Exception as e:
            print(f"🔄 SEQ ERROR: Error preparing sequence data: {e}")
            print(f"🔄 SEQ ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"🔄 SEQ ERROR: Error preparing sequence data: {e}")
            logger.error(f"🔄 SEQ ERROR TRACEBACK:\n{traceback.format_exc()}")
            return None, None
    
    def _prepare_lstm_input(self, X):
        """
        Prepare input data for LSTM prediction.
        
        Args:
            X: DataFrame with features
            
        Returns:
            numpy.ndarray: Prepared LSTM input or None if preparation fails
        """
        if not TENSORFLOW_AVAILABLE or self.lstm_model is None or self.lstm_scaler is None:
            print("🧠 LSTM INPUT: TensorFlow/model/scaler not available")
            logger.error("🧠 LSTM INPUT: TensorFlow/model/scaler not available")
            return None
            
        try:
            print("🧠 LSTM INPUT: _prepare_lstm_input called")
            logger.error("🧠 LSTM INPUT: _prepare_lstm_input called")
            
            # Check if we have stored LSTM feature columns
            lstm_feature_cols = self.lstm_feature_cols
            
            # If not, fall back to all feature columns
            if not lstm_feature_cols:
                lstm_feature_cols = self.all_feature_cols
                print("🧠 LSTM INPUT: Using all_feature_cols as fallback")
                logger.error("🧠 LSTM INPUT: Using all_feature_cols as fallback")
            
            # Check which columns are available
            available_cols = [col for col in lstm_feature_cols if col in X.columns]
            
            print(f"🧠 LSTM INPUT: {len(available_cols)}/{len(lstm_feature_cols)} LSTM features available")
            logger.error(f"🧠 LSTM INPUT: {len(available_cols)}/{len(lstm_feature_cols)} LSTM features available")
            
            # If too few columns are available, return None
            if len(available_cols) < 0.5 * len(lstm_feature_cols):
                print(f"🧠 LSTM INPUT: Too few LSTM features available: {len(available_cols)}/{len(lstm_feature_cols)}")
                logger.error(f"🧠 LSTM INPUT: Too few LSTM features available: {len(available_cols)}/{len(lstm_feature_cols)}")
                return None
                
            # Use available columns
            X_subset = X[available_cols].copy()
            
            # Handle missing values
            X_subset = X_subset.fillna(method='ffill')
            X_subset = X_subset.fillna(method='bfill')
            X_subset = X_subset.fillna(0)  # Fill any remaining NaNs with 0
            
            print("🧠 LSTM INPUT: Handled missing values in input data")
            logger.error("🧠 LSTM INPUT: Handled missing values in input data")
            
            # Scale features
            X_scaled = self.lstm_scaler.transform(X_subset)
            print("🧠 LSTM INPUT: Scaled input features")
            logger.error("🧠 LSTM INPUT: Scaled input features")
            
            # For single sample prediction
            if len(X_subset) == 1:
                print("🧠 LSTM INPUT: Creating sequence for single sample")
                logger.error("🧠 LSTM INPUT: Creating sequence for single sample")
                # Create a sequence with repeated values
                sequence = np.repeat(X_scaled, self.lstm_sequence_length, axis=0)
                sequence = sequence.reshape(1, self.lstm_sequence_length, -1)
                print(f"🧠 LSTM INPUT: Created single sample sequence with shape {sequence.shape}")
                logger.error(f"🧠 LSTM INPUT: Created single sample sequence with shape {sequence.shape}")
                return sequence
                
            # For multiple samples (during training/evaluation)
            else:
                print("🧠 LSTM INPUT: Creating sequences for multiple samples")
                logger.error("🧠 LSTM INPUT: Creating sequences for multiple samples")
                # Create sequences from the dataset
                sequences = []
                for i in range(len(X_scaled) - self.lstm_sequence_length + 1):
                    seq = X_scaled[i:i+self.lstm_sequence_length]
                    sequences.append(seq)
                
                # If we need more sequences to match X length
                if len(sequences) < len(X_scaled):
                    padding_needed = len(X_scaled) - len(sequences)
                    print(f"🧠 LSTM INPUT: Adding {padding_needed} padding sequences")
                    logger.error(f"🧠 LSTM INPUT: Adding {padding_needed} padding sequences")
                    
                    # Repeat the first sequence with padding
                    for i in range(padding_needed):
                        # Create padding by repeating the first row
                        padding = np.repeat(X_scaled[0:1], self.lstm_sequence_length - 1, axis=0)
                        # Add the first i+1 actual values
                        seq = np.vstack([padding[:-(i+1)], X_scaled[:(i+1)]])
                        sequences.insert(0, seq)
                
                sequences_array = np.array(sequences)
                print(f"🧠 LSTM INPUT: Created {len(sequences)} sequences with shape {sequences_array.shape}")
                logger.error(f"🧠 LSTM INPUT: Created {len(sequences)} sequences with shape {sequences_array.shape}")
                return sequences_array
        
        except Exception as e:
            print(f"🧠 LSTM INPUT ERROR: Error preparing LSTM input: {e}")
            print(f"🧠 LSTM INPUT ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"🧠 LSTM INPUT ERROR: Error preparing LSTM input: {e}")
            logger.error(f"🧠 LSTM INPUT ERROR TRACEBACK:\n{traceback.format_exc()}")
            return None

    def predict_with_ensemble(self, X):
        """
        Make a prediction using the full ensemble model.
        
        Args:
            X: DataFrame with features
            
        Returns:
            tuple: (probability, prediction)
        """
        try:
            # Debug logging
            print("🔮 PREDICT: predict_with_ensemble called")
            logger.error("🔮 PREDICT: predict_with_ensemble called")
            
            print(f"🔮 PREDICT: Input shape: {X.shape if hasattr(X, 'shape') else 'unknown'}")
            logger.error(f"🔮 PREDICT: Input shape: {X.shape if hasattr(X, 'shape') else 'unknown'}")
            
            print(f"🔮 PREDICT: Meta-model exists: {self.meta_model is not None}")
            logger.error(f"🔮 PREDICT: Meta-model exists: {self.meta_model is not None}")
            
            print(f"🔮 PREDICT: Meta-scaler exists: {self.meta_scaler is not None}")
            logger.error(f"🔮 PREDICT: Meta-scaler exists: {self.meta_scaler is not None}")
            
            # Sanity check on input
            if X is None or len(X) == 0:
                print("🔮 PREDICT: Empty input for prediction")
                logger.error("🔮 PREDICT: Empty input for prediction")
                return 0.5, 0
            
            # Handle missing features
            if not self.all_feature_cols:
                print("🔮 PREDICT: No feature columns defined")
                logger.error("🔮 PREDICT: No feature columns defined")
                return 0.5, 0
                
            # Make sure X has all required columns with appropriate handling for missing columns
            missing_cols = [col for col in self.all_feature_cols if col not in X.columns]
            if missing_cols:
                print(f"🔮 PREDICT: Missing {len(missing_cols)}/{len(self.all_feature_cols)} feature columns. Using defaults.")
                logger.error(f"🔮 PREDICT: Missing {len(missing_cols)}/{len(self.all_feature_cols)} feature columns. Using defaults.")
                
                # Add missing columns with default values
                for col in missing_cols:
                    X[col] = 0.0
                    
                print(f"🔮 PREDICT: Added missing columns with default values")
                logger.error(f"🔮 PREDICT: Added missing columns with default values")
            
            # Initialize dictionary to store model predictions
            model_predictions = {}
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                print(f"🔮 PREDICT: Getting prediction from {model_name}")
                logger.error(f"🔮 PREDICT: Getting prediction from {model_name}")
                
                # Get feature subset and scaler for this model
                features = self.feature_subsets.get(model_name, [])
                scaler = self.scalers.get(model_name)
                
                if not features or scaler is None:
                    print(f"🔮 PREDICT: Missing features or scaler for {model_name}")
                    logger.error(f"🔮 PREDICT: Missing features or scaler for {model_name}")
                    continue
                
                # Check if all required features exist
                missing_features = [f for f in features if f not in X.columns]
                if missing_features:
                    print(f"🔮 PREDICT: Missing features for {model_name}: {missing_features}")
                    logger.error(f"🔮 PREDICT: Missing features for {model_name}: {missing_features}")
                    # Add missing features with zeros
                    for f in missing_features:
                        X[f] = 0.0
                    print(f"🔮 PREDICT: Added missing features with default values for {model_name}")
                    logger.error(f"🔮 PREDICT: Added missing features with default values for {model_name}")
                
            try:
                    # Scale features
                    X_scaled = scaler.transform(X[features])
                    
                    # Get probability prediction
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_scaled)[:, 1][0]  # Get probability of positive class
                        print(f"🔮 PREDICT: {model_name} probability: {prob:.4f}")
                        logger.error(f"🔮 PREDICT: {model_name} probability: {prob:.4f}")
                    else:
                        prob = float(model.predict(X_scaled)[0])  # Convert prediction to probability
                        print(f"🔮 PREDICT: {model_name} prediction: {prob}")
                        logger.error(f"🔮 PREDICT: {model_name} prediction: {prob}")
                    
                    model_predictions[f"{model_name}_prob"] = prob
            except Exception as e:
                    print(f"🔮 PREDICT: Error getting prediction from {model_name}: {e}")
                    print(f"🔮 PREDICT ERROR TRACEBACK:\n{traceback.format_exc()}")
                    logger.error(f"🔮 PREDICT: Error getting prediction from {model_name}: {e}")
                    logger.error(f"🔮 PREDICT ERROR TRACEBACK:\n{traceback.format_exc()}")
                    model_predictions[f"{model_name}_prob"] = 0.5  # Default value
                    print(f"🔮 PREDICT: Using default value 0.5 for {model_name}")
                    logger.error(f"🔮 PREDICT: Using default value 0.5 for {model_name}")
            
            # Add LSTM prediction if available
            if TENSORFLOW_AVAILABLE and self.lstm_model is not None:
                print("🔮 PREDICT: Getting LSTM prediction")
                logger.error("🔮 PREDICT: Getting LSTM prediction")
                try:
                    lstm_X = self._prepare_lstm_input(X)
                    if lstm_X is not None:
                        lstm_prob = float(self.lstm_model.predict(lstm_X).flatten()[0])
                        model_predictions['lstm_prob'] = lstm_prob
                        print(f"🔮 PREDICT: LSTM probability: {lstm_prob:.4f}")
                        logger.error(f"🔮 PREDICT: LSTM probability: {lstm_prob:.4f}")
                    else:
                        model_predictions['lstm_prob'] = 0.5  # Default value
                        print("🔮 PREDICT: Could not prepare LSTM input, using default value 0.5")
                        logger.error("🔮 PREDICT: Could not prepare LSTM input, using default value 0.5")
                except Exception as e:
                    print(f"🔮 PREDICT: Error getting LSTM prediction: {e}")
                    print(f"🔮 PREDICT ERROR TRACEBACK:\n{traceback.format_exc()}")
                    logger.error(f"🔮 PREDICT: Error getting LSTM prediction: {e}")
                    logger.error(f"🔮 PREDICT ERROR TRACEBACK:\n{traceback.format_exc()}")
                    model_predictions['lstm_prob'] = 0.5  # Default value
                    print("🔮 PREDICT: Using default value 0.5 for LSTM")
                    logger.error("🔮 PREDICT: Using default value 0.5 for LSTM")
            else:
                model_predictions['lstm_prob'] = 0.5  # Default value
                print("🔮 PREDICT: No LSTM model available, using default value 0.5")
                logger.error("🔮 PREDICT: No LSTM model available, using default value 0.5")
            
            # Debug logging of meta features
            print(f"🔮 PREDICT: Created meta features with keys: {list(model_predictions.keys())}")
            logger.error(f"🔮 PREDICT: Created meta features with keys: {list(model_predictions.keys())}")
            
            # If no models could make predictions, return default values
            if not model_predictions:
                print("🔮 PREDICT: No valid predictions from any model")
                logger.error("🔮 PREDICT: No valid predictions from any model")
                return 0.5, 0
            
            # If meta-model is available, use it to combine predictions
            if self.meta_model is not None and self.meta_scaler is not None and self.meta_feature_names:
                try:
                    print("🔮 PREDICT: Using meta-model for final prediction")
                    logger.error("🔮 PREDICT: Using meta-model for final prediction")
                    
                    # Create meta-features DataFrame with expected columns
                    meta_df = pd.DataFrame([model_predictions])
                    
                    # Debug log expected meta feature names
                    print(f"🔮 PREDICT: Expected meta feature names: {self.meta_feature_names}")
                    logger.error(f"🔮 PREDICT: Expected meta feature names: {self.meta_feature_names}")
                    print(f"🔮 PREDICT: Meta DataFrame columns: {meta_df.columns.tolist()}")
                    logger.error(f"🔮 PREDICT: Meta DataFrame columns: {meta_df.columns.tolist()}")
                    
                    # Ensure all expected features are present
                    for col in self.meta_feature_names:
                        if col not in meta_df.columns:
                            meta_df[col] = 0.5  # Add missing columns with default value
                            print(f"🔮 PREDICT: Added missing meta feature '{col}' with default value 0.5")
                            logger.error(f"🔮 PREDICT: Added missing meta feature '{col}' with default value 0.5")
                    
                    # Keep only the expected columns in the right order
                    meta_df = meta_df[self.meta_feature_names]
                    
                    # Critical check for lstm_prob
                    if 'lstm_prob' not in meta_df.columns:
                        print("🔮 PREDICT: Critical error: 'lstm_prob' column is missing!")
                        logger.error("🔮 PREDICT: Critical error: 'lstm_prob' column is missing!")
                        meta_df['lstm_prob'] = 0.5
                        print("🔮 PREDICT: Added missing 'lstm_prob' column with default value 0.5")
                        logger.error("🔮 PREDICT: Added missing 'lstm_prob' column with default value 0.5")
                    
                    # Scale meta-features
                    meta_scaled = self.meta_scaler.transform(meta_df)
                    print("🔮 PREDICT: Scaled meta-features")
                    logger.error("🔮 PREDICT: Scaled meta-features")
                    
                    # Use meta-model to get final prediction
                    if hasattr(self.meta_model, 'predict_proba'):
                        final_prob = float(self.meta_model.predict_proba(meta_scaled)[:, 1][0])
                        print(f"🔮 PREDICT: Meta-model probability: {final_prob:.4f}")
                        logger.error(f"🔮 PREDICT: Meta-model probability: {final_prob:.4f}")
                    else:
                        final_prob = float(self.meta_model.predict(meta_scaled)[0])
                        print(f"🔮 PREDICT: Meta-model prediction: {final_prob}")
                        logger.error(f"🔮 PREDICT: Meta-model prediction: {final_prob}")
                except Exception as e:
                    print(f"🔮 PREDICT: Error in meta-model prediction: {e}")
                    print(f"🔮 PREDICT ERROR TRACEBACK:\n{traceback.format_exc()}")
                    logger.error(f"🔮 PREDICT: Error in meta-model prediction: {e}")
                    logger.error(f"🔮 PREDICT ERROR TRACEBACK:\n{traceback.format_exc()}")
                    
                    # Fall back to simple averaging
                    final_prob = sum(model_predictions.values()) / len(model_predictions)
                    print(f"🔮 PREDICT: Using average of model predictions as fallback: {final_prob:.4f}")
                    logger.error(f"🔮 PREDICT: Using average of model predictions as fallback: {final_prob:.4f}")
            else:
                # Simple averaging if no meta-model
                final_prob = sum(model_predictions.values()) / len(model_predictions)
                print(f"🔮 PREDICT: No meta-model available, using average: {final_prob:.4f}")
                logger.error(f"🔮 PREDICT: No meta-model available, using average: {final_prob:.4f}")
            
            # Convert probability to class label (0 or 1)
            prediction = 1 if final_prob >= 0.5 else 0
            
            print(f"🔮 PREDICT: Final prediction: {prediction} with probability {final_prob:.4f}")
            logger.error(f"🔮 PREDICT: Final prediction: {prediction} with probability {final_prob:.4f}")
            
            return final_prob, prediction
                
        except Exception as e:
            print(f"🔮 PREDICT ERROR: Error in ensemble prediction: {e}")
            print(f"🔮 PREDICT ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"🔮 PREDICT ERROR: Error in ensemble prediction: {e}")
            logger.error(f"🔮 PREDICT ERROR TRACEBACK:\n{traceback.format_exc()}")
            return 0.5, 0  # Default values
    
    def _get_top_features(self, n=10):
        """
        Get top n features by importance across all models.
        
        Args:
            n: Number of top features to return
            
        Returns:
            list: Top n feature names
        """
        try:
            print(f"📊 TOP: _get_top_features called for top {n} features")
            logger.error(f"📊 TOP: _get_top_features called for top {n} features")
            
            if not self.feature_importance:
                print("📊 TOP: No feature importance data available")
                logger.error("📊 TOP: No feature importance data available")
                return []
                
            # Combine feature importance from all models
            combined_importance = defaultdict(float)
            
            # Sum importance across models
            for model_name, importances in self.feature_importance.items():
                print(f"📊 TOP: Adding importance from {model_name}")
                logger.error(f"📊 TOP: Adding importance from {model_name}")
                for feature, importance in importances.items():
                    combined_importance[feature] += importance
            
            # Sort features by importance
            sorted_features = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Get top n features
            top_n = [f[0] for f in sorted_features[:n]]
            
            print(f"📊 TOP: Top {len(top_n)} features: {top_n}")
            logger.error(f"📊 TOP: Top {len(top_n)} features: {top_n}")
            
            # Return top n feature names
            return top_n
            
        except Exception as e:
            print(f"📊 TOP ERROR: Error getting top features: {e}")
            print(f"📊 TOP ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"📊 TOP ERROR: Error getting top features: {e}")
            logger.error(f"📊 TOP ERROR TRACEBACK:\n{traceback.format_exc()}")
            return []
            
    def get_model_info(self):
        """
        Get information about the trained models.
        
        Returns:
            dict: Model information
        """
        try:
            print("ℹ️ INFO: get_model_info called")
            logger.error("ℹ️ INFO: get_model_info called")
            
            info = {
                'base_models': list(self.models.keys()),
                'meta_model': type(self.meta_model).__name__ if self.meta_model else None,
                'lstm_model': 'LSTM' if self.lstm_model is not None else None,
                'feature_subsets': {k: len(v) for k, v in self.feature_subsets.items()},
                'total_features': len(self.all_feature_cols) if self.all_feature_cols else 0,
                'metrics': self.model_metrics,
                'top_features': self._get_top_features(10)
            }
            
            print(f"ℹ️ INFO: Model info collected successfully with {len(info['base_models'])} base models")
            logger.error(f"ℹ️ INFO: Model info collected successfully with {len(info['base_models'])} base models")
            
            return info
            
        except Exception as e:
            print(f"ℹ️ INFO ERROR: Error getting model info: {e}")
            print(f"ℹ️ INFO ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"ℹ️ INFO ERROR: Error getting model info: {e}")
            logger.error(f"ℹ️ INFO ERROR TRACEBACK:\n{traceback.format_exc()}")
            return {'error': str(e)}
            
    def predict_trend(self, data, symbol=None):
        """
        Predict trend (increase/decrease) for the given data.
        
        Args:
            data: DataFrame with market data
            symbol: Optional symbol for logging
            
        Returns:
            dict: Prediction results
        """
        try:
            print(f"📈 TREND: predict_trend called for {symbol if symbol else 'unknown'}")
            logger.error(f"📈 TREND: predict_trend called for {symbol if symbol else 'unknown'}")
            
            # Run the debug method to help diagnose issues
            self.debug_meta_features()
            
            if data is None or len(data) == 0:
                print("📈 TREND: No data provided")
                logger.error("📈 TREND: No data provided")
                return {'error': 'No data provided'}
                
            # Make prediction
            probability, prediction = self.predict_with_ensemble(data)
            
            # Convert to trend
            trend = "INCREASE" if prediction == 1 else "DECREASE"
            
            # Log prediction
            print(f"📈 TREND: Prediction for {symbol if symbol else 'unknown'}: {trend}")
            print(f"📈 TREND: Confidence: {probability:.4f}")
            logger.error(f"📈 TREND: Prediction for {symbol if symbol else 'unknown'}: {trend}")
            logger.error(f"📈 TREND: Confidence: {probability:.4f}")
            
            result = {
                'symbol': symbol,
                'trend': trend,
                'probability': float(probability),
                'prediction': int(prediction),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"📈 TREND: Returning prediction result: {result}")
            logger.error(f"📈 TREND: Returning prediction result: {result}")
            
            return result
            
        except Exception as e:
            print(f"📈 TREND ERROR: Error predicting trend: {e}")
            print(f"📈 TREND ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"📈 TREND ERROR: Error predicting trend: {e}")
            logger.error(f"📈 TREND ERROR TRACEBACK:\n{traceback.format_exc()}")
            return {'error': str(e)}
            
    def calculate_feature_correlation(self, data, top_n=20):
        """
        Calculate correlation between features and target.
        
        Args:
            data: DataFrame with features and target
            top_n: Number of top correlated features to return
            
        Returns:
            dict: Correlation information
        """
        try:
            print(f"🔄 CORR: calculate_feature_correlation called for top {top_n} features")
            logger.error(f"🔄 CORR: calculate_feature_correlation called for top {top_n} features")
            
            if 'Label' not in data.columns:
                print("🔄 CORR: Label column not found in data")
                logger.error("🔄 CORR: Label column not found in data")
                return {'error': 'Label column not found'}
                
            # Get feature columns
            feature_cols = [col for col in data.columns if col in self.all_feature_cols]
            
            if not feature_cols:
                print("🔄 CORR: No matching feature columns found")
                logger.error("🔄 CORR: No matching feature columns found")
                return {'error': 'No matching feature columns found'}
                
            # Calculate correlation with Label
            correlations = {}
            for col in feature_cols:
                if data[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    corr = data[col].corr(data['Label'])
                    if not np.isnan(corr):
                        correlations[col] = corr
            
            # Sort by absolute correlation
            sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Get top N correlated features
            top_correlated = sorted_corr[:top_n]
            
            print(f"🔄 CORR: Found {len(top_correlated)} correlated features")
            logger.error(f"🔄 CORR: Found {len(top_correlated)} correlated features")
            
            result = {
                'top_correlated_features': dict(top_correlated),
                'positive_correlated': dict(sorted(
                    [(k, v) for k, v in sorted_corr if v > 0],
                    key=lambda x: x[1], reverse=True
                )[:top_n]),
                'negative_correlated': dict(sorted(
                    [(k, v) for k, v in sorted_corr if v < 0],
                    key=lambda x: x[1]
                )[:top_n])
            }
            
            print(f"🔄 CORR: Calculated correlation results successfully")
            logger.error(f"🔄 CORR: Calculated correlation results successfully")
            
            return result
            
        except Exception as e:
            print(f"🔄 CORR ERROR: Error calculating feature correlation: {e}")
            print(f"🔄 CORR ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"🔄 CORR ERROR: Error calculating feature correlation: {e}")
            logger.error(f"🔄 CORR ERROR TRACEBACK:\n{traceback.format_exc()}")
            return {'error': str(e)}
    
    def detect_market_regime(self, data, n_regimes=None):
        """
        Detect market regime from data.
        
        Args:
            data: DataFrame with market data
            n_regimes: Optional override for number of regimes
            
        Returns:
            dict: Detected market regime information
        """
        try:
            print("🔍 REGIME: detect_market_regime called")
            logger.error("🔍 REGIME: detect_market_regime called")
            
            if n_regimes is None:
                n_regimes = self.n_regimes
            
            # Simple regime detection based on volatility and trend
            if 'Close' not in data.columns:
                print("🔍 REGIME: Close column not found in data")
                logger.error("🔍 REGIME: Close column not found in data")
                return {'error': 'Close column not found'}
            
            # Calculate returns
            returns = data['Close'].pct_change().dropna()
            
            if len(returns) < 10:
                print("🔍 REGIME: Not enough data for regime detection")
                logger.error("🔍 REGIME: Not enough data for regime detection")
                return {'error': 'Not enough data for regime detection'}
            
            # Calculate volatility (20-day rolling standard deviation)
            volatility = returns.rolling(window=20).std().dropna()
            
            # Calculate trend (20-day returns)
            trend = returns.rolling(window=20).mean().dropna()
            
            if len(volatility) == 0 or len(trend) == 0:
                print("🔍 REGIME: Not enough data for volatility/trend calculation")
                logger.error("🔍 REGIME: Not enough data for volatility/trend calculation")
                return {'error': 'Not enough data for volatility/trend calculation'}
            
            # Get current values
            current_volatility = volatility.iloc[-1]
            current_trend = trend.iloc[-1]
            
            # Categorize regime
            if current_trend > 0 and current_volatility < volatility.quantile(0.7):
                regime = "Bull Market (Low Volatility)"
                regime_id = 0
            elif current_trend > 0 and current_volatility >= volatility.quantile(0.7):
                regime = "Bull Market (High Volatility)"
                regime_id = 1
            elif current_trend <= 0 and current_volatility < volatility.quantile(0.7):
                regime = "Bear Market (Low Volatility)"
                regime_id = 2
            else:
                regime = "Bear Market (High Volatility)"
                regime_id = 3
            
            print(f"🔍 REGIME: Detected regime: {regime}")
            logger.error(f"🔍 REGIME: Detected regime: {regime}")
            
            result = {
                'regime': regime,
                'regime_id': regime_id,
                'volatility': float(current_volatility),
                'trend': float(current_trend),
                'avg_volatility': float(volatility.mean()),
                'max_volatility': float(volatility.max()),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"🔍 REGIME ERROR: Error detecting market regime: {e}")
            print(f"🔍 REGIME ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"🔍 REGIME ERROR: Error detecting market regime: {e}")
            logger.error(f"🔍 REGIME ERROR TRACEBACK:\n{traceback.format_exc()}")
            return {'error': str(e)}
    
    def analyze_feature_importance(self, data=None, top_n=20):
        """
        Analyze feature importance across models.
        
        Args:
            data: Optional DataFrame for calculating correlation with target
            top_n: Number of top features to return
            
        Returns:
            dict: Feature importance analysis
        """
        try:
            print(f"📊 FEATURE: analyze_feature_importance called for top {top_n} features")
            logger.error(f"📊 FEATURE: analyze_feature_importance called for top {top_n} features")
            
            # Get top features from model importance
            top_features = self._get_top_features(top_n)
            
            # Calculate correlation if data is provided
            correlation_results = None
            if data is not None and 'Label' in data.columns:
                correlation_results = self.calculate_feature_correlation(data, top_n)
            
            # Get feature importance per model
            per_model_importance = {}
            for model_name, importances in self.feature_importance.items():
                # Sort by importance
                sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                # Get top N
                per_model_importance[model_name] = dict(sorted_importances[:top_n])
            
            print(f"📊 FEATURE: Analyzed feature importance across {len(per_model_importance)} models")
            logger.error(f"📊 FEATURE: Analyzed feature importance across {len(per_model_importance)} models")
            
            result = {
                'top_features': top_features,
                'per_model_importance': per_model_importance,
                'correlation_results': correlation_results,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"📊 FEATURE ERROR: Error analyzing feature importance: {e}")
            print(f"📊 FEATURE ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"📊 FEATURE ERROR: Error analyzing feature importance: {e}")
            logger.error(f"📊 FEATURE ERROR TRACEBACK:\n{traceback.format_exc()}")
            return {'error': str(e)}
    
    def get_model_metrics(self):
        """
        Get metrics for all models.
        
        Returns:
            dict: Model metrics
        """
        try:
            print("📏 METRICS: get_model_metrics called")
            logger.error("📏 METRICS: get_model_metrics called")
            
            if not self.model_metrics:
                print("📏 METRICS: No model metrics available")
                logger.error("📏 METRICS: No model metrics available")
                return {'error': 'No model metrics available'}
            
            # Enhanced metrics summary
            metrics_summary = {
                'meta_learner': self.model_metrics.get('meta_learner', {}),
                'ensemble': self.model_metrics.get('ensemble', {}),
                'lstm': self.model_metrics.get('lstm', {}),
                'base_models': {
                    model_name: metrics 
                    for model_name, metrics in self.model_metrics.items() 
                    if model_name not in ['meta_learner', 'ensemble', 'lstm']
                }
            }
            
            # Add overall best model by different metrics
            if len(self.model_metrics) > 0:
                best_models = {}
                
                # Find best model by each metric
                for metric in ['accuracy', 'f1_score', 'f2_score', 'auc', 'precision', 'recall']:
                    best_model = max(self.model_metrics.items(), 
                                    key=lambda x: x[1].get(metric, 0) 
                                    if isinstance(x[1], dict) else 0)
                    best_models[f'best_{metric}'] = {
                        'model': best_model[0],
                        'value': best_model[1].get(metric, 0) if isinstance(best_model[1], dict) else 0
                    }
                
                metrics_summary['best_models'] = best_models
            
            print(f"📏 METRICS: Retrieved metrics for {len(self.model_metrics)} models")
            logger.error(f"📏 METRICS: Retrieved metrics for {len(self.model_metrics)} models")
            
            return metrics_summary
            
        except Exception as e:
            print(f"📏 METRICS ERROR: Error getting model metrics: {e}")
            print(f"📏 METRICS ERROR TRACEBACK:\n{traceback.format_exc()}")
            logger.error(f"📏 METRICS ERROR: Error getting model metrics: {e}")
            logger.error(f"📏 METRICS ERROR TRACEBACK:\n{traceback.format_exc()}")
            return {'error': str(e)}