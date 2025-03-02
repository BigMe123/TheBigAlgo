import os
import joblib
import logging
import platform
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union

# Set up logging first to ensure error messages are captured properly
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Machine learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, fbeta_score, roc_auc_score, accuracy_score
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold

# Deep learning imports (if tensorflow is available)
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras

    Sequential = keras.Sequential
    Dense = keras.layers.Dense
    LSTM = keras.layers.LSTM
    Dropout = keras.layers.Dropout
    BatchNormalization = keras.layers.BatchNormalization
    Adam = keras.optimizers.Adam
    EarlyStopping = keras.callbacks.EarlyStopping
    ModelCheckpoint = keras.callbacks.ModelCheckpoint
    load_model = keras.models.load_model
    save_model = keras.models.save_model

    TENSORFLOW_AVAILABLE = True
except ImportError:
    if platform.processor() == 'arm':
        logger.error("TensorFlow not found. For Apple Silicon Macs, try:\npip install tensorflow-macos tensorflow-metal")
    else:
        logger.error("TensorFlow not found. Install with: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow could not be imported. Deep learning features will be disabled.")


class MLTradingPredictor:
    """
    Ensemble model that combines multiple ML models using a meta-learner approach.
    """

    def __init__(self, model_dir="./ensemble_models"):
        logger.debug("Entering MLTradingPredictor.__init__")
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Sub-models in the ensemble
        self.models = {}
        self.scalers = {}
        self.feature_subsets = {}

        # Meta-learner model, its scaler, and expected meta-feature names.
        self.meta_model = None
        self.meta_scaler = None
        self.meta_feature_names = None  # Should be a list of strings

        # Track feature importance across all models
        self.feature_importance = {}

        # Sub-model performance metrics
        self.model_metrics = {}

        # LSTM model (TensorFlow required)
        self.lstm_model = None
        self.lstm_sequence_length = 10  # Default sequence length for LSTM

        # Load models if they exist
        self.load_models()
        logger.debug("Exiting MLTradingPredictor.__init__")

    def load_models(self):
        logger.debug("Entering MLTradingPredictor.load_models")
        try:
            if not os.path.exists(self.model_dir):
                logger.info(f"Model directory {self.model_dir} not found. Will create new models.")
                return

            registry_path = os.path.join(self.model_dir, "model_registry.joblib")
            if not os.path.exists(registry_path):
                logger.info("Model registry not found. Will create new models.")
                return

            registry = joblib.load(registry_path)
            logger.debug(f"Registry loaded: {registry}")

            for model_name, model_info in registry.get('models', {}).items():
                model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
                scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.joblib")
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[model_name] = joblib.load(model_path)
                    self.scalers[model_name] = joblib.load(scaler_path)
                    self.feature_subsets[model_name] = model_info.get('features', [])
                    logger.info(f"Loaded model '{model_name}' with features: {self.feature_subsets[model_name]}")

            meta_model_path = os.path.join(self.model_dir, "meta_model.joblib")
            meta_scaler_path = os.path.join(self.model_dir, "meta_scaler.joblib")
            if os.path.exists(meta_model_path) and os.path.exists(meta_scaler_path):
                self.meta_model = joblib.load(meta_model_path)
                self.meta_scaler = joblib.load(meta_scaler_path)
                meta_feature_names_path = os.path.join(self.model_dir, "meta_feature_names.joblib")
                if os.path.exists(meta_feature_names_path):
                    self.meta_feature_names = joblib.load(meta_feature_names_path)
                    self.meta_feature_names = [str(x) for x in self.meta_feature_names]
                logger.info("Loaded meta-model")
            else:
                logger.info("No meta-model found.")

            importance_path = os.path.join(self.model_dir, "feature_importance.joblib")
            if os.path.exists(importance_path):
                self.feature_importance = joblib.load(importance_path)
                logger.debug(f"Loaded feature importance: {self.feature_importance}")

            metrics_path = os.path.join(self.model_dir, "model_metrics.joblib")
            if os.path.exists(metrics_path):
                self.model_metrics = joblib.load(metrics_path)
                logger.debug(f"Loaded model metrics: {self.model_metrics}")

            if TENSORFLOW_AVAILABLE:
                lstm_path = os.path.join(self.model_dir, "lstm_model.keras")
                if os.path.exists(lstm_path):
                    try:
                        self.lstm_model = load_model(lstm_path)
                        logger.info("Loaded LSTM model")
                    except Exception as e:
                        logger.warning(f"Error loading LSTM model: {e}")

            logger.info(f"Successfully loaded {len(self.models)} base models and meta-model: {self.meta_model is not None}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models = {}
            self.scalers = {}
            self.feature_subsets = {}
            self.meta_model = None
            self.meta_scaler = None
        logger.debug("Exiting MLTradingPredictor.load_models")

    def save_models(self):
        logger.debug("Entering MLTradingPredictor.save_models")
        try:
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

            for model_name, model in self.models.items():
                model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
                scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.joblib")
                joblib.dump(model, model_path)
                joblib.dump(self.scalers[model_name], scaler_path)
                logger.debug(f"Saved model '{model_name}' and its scaler.")

            if self.meta_model is not None and self.meta_scaler is not None:
                meta_model_path = os.path.join(self.model_dir, "meta_model.joblib")
                meta_scaler_path = os.path.join(self.model_dir, "meta_scaler.joblib")
                joblib.dump(self.meta_model, meta_model_path)
                joblib.dump(self.meta_scaler, meta_scaler_path)
                joblib.dump(self.meta_feature_names, os.path.join(self.model_dir, "meta_feature_names.joblib"))
                logger.debug("Saved meta-model and its scaler.")

            if self.feature_importance:
                importance_path = os.path.join(self.model_dir, "feature_importance.joblib")
                joblib.dump(self.feature_importance, importance_path)
                logger.debug("Saved feature importance.")

            if self.model_metrics:
                metrics_path = os.path.join(self.model_dir, "model_metrics.joblib")
                joblib.dump(self.model_metrics, metrics_path)
                logger.debug("Saved model metrics.")

            if TENSORFLOW_AVAILABLE and self.lstm_model is not None:
                lstm_path = os.path.join(self.model_dir, "lstm_model.keras")
                self.lstm_model.save(lstm_path)
                logger.debug("Saved LSTM model.")

            registry = {
                'last_updated': datetime.now(),
                'models': {
                    model_name: {
                        'type': type(model).__name__,
                        'features': self.feature_subsets.get(model_name, [])
                    } for model_name, model in self.models.items()
                },
                'meta_model': type(self.meta_model).__name__ if self.meta_model else None,
                'lstm_model': 'LSTM' if self.lstm_model is not None else None
            }
            registry_path = os.path.join(self.model_dir, "model_registry.joblib")
            joblib.dump(registry, registry_path)
            logger.info(f"Successfully saved all models to {self.model_dir}")
            logger.debug("Exiting MLTradingPredictor.save_models with success")
            return True
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            logger.debug("Exiting MLTradingPredictor.save_models with error")
            return False

    def _fallback_prediction(self, X):
        logger.debug("Entering _fallback_prediction")
        logger.debug(f"Input X shape: {X.shape}, columns: {X.columns.tolist() if isinstance(X, pd.DataFrame) else 'N/A'}")
        probas = []
        for model_name, model in self.models.items():
            features = self.feature_subsets.get(model_name)
            if features is None or not all(f in X.columns for f in features):
                logger.warning(f"Skipping model '{model_name}' in fallback due to missing features.")
                continue
            scaler = self.scalers.get(model_name)
            if scaler is None:
                logger.warning(f"No scaler found for model '{model_name}' in fallback.")
                continue
            X_scaled = scaler.transform(X[features])
            logger.debug(f"Model '{model_name}': X_scaled shape: {X_scaled.shape}")
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_scaled)[0, 1]
                    logger.debug(f"Model '{model_name}' fallback proba: {proba}")
                except Exception as e:
                    logger.error(f"Error in fallback prediction for model '{model_name}': {e}")
                    continue
            else:
                proba = float(model.predict(X_scaled)[0])
                logger.debug(f"Model '{model_name}' fallback prediction (non-proba): {proba}")
            probas.append(proba)
        if probas:
            avg_proba = sum(probas) / len(probas)
            confidence = abs(avg_proba - 0.5) * 2
            logger.debug(f"Fallback prediction: {avg_proba} with confidence {confidence}")
            logger.debug("Exiting _fallback_prediction with success")
            return avg_proba, confidence
        else:
            logger.warning("No valid base model predictions available in fallback; using default probability 0.5")
            logger.debug("Exiting _fallback_prediction with default")
            return 0.5, 0

    def predict_with_ensemble(self, X):
        logger.debug("Entering predict_with_ensemble")
        logger.debug(f"Input X type: {type(X)}; shape: {X.shape if hasattr(X, 'shape') else 'N/A'}; columns: {X.columns.tolist() if hasattr(X, 'columns') else 'N/A'}")
        try:
            meta_features = pd.DataFrame()
            for model_name, model in self.models.items():
                features = self.feature_subsets.get(model_name)
                if features is None:
                    logger.warning(f"Feature subset for model '{model_name}' not found; skipping.")
                    continue
                missing_features = [f for f in features if f not in X.columns]
                if missing_features:
                    logger.warning(f"Missing features for model '{model_name}': {missing_features}; skipping.")
                    continue
                scaler = self.scalers[model_name]
                X_scaled = scaler.transform(X[features])
                logger.debug(f"Model '{model_name}': features: {features}; X_scaled shape: {X_scaled.shape}")
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)[0, 1]
                else:
                    proba = float(model.predict(X_scaled)[0])
                logger.debug(f"Model '{model_name}' prediction: {proba}")
                meta_features[f"{model_name}_prob"] = np.array(proba).reshape(-1)
            logger.debug(f"Meta-features before reindex:\n{meta_features.head()}\nShape: {meta_features.shape}")
            if self.meta_feature_names is None:
                expected_cols = meta_features.columns.tolist()
                logger.debug("Meta-feature names not set; using dynamic columns: %s", expected_cols)
            else:
                expected_cols = [str(col) for col in self.meta_feature_names]
                logger.debug("Using stored meta_feature_names: %s", expected_cols)
            logger.debug(f"Meta-features before reindex shape: {meta_features.shape}; expected columns: {expected_cols}")
            meta_features = meta_features.reindex(columns=expected_cols, fill_value=0.5)
            logger.debug(f"Meta-features after reindex shape: {meta_features.shape}; columns: {meta_features.columns.tolist()}")
            if self.meta_model is None or self.meta_scaler is None:
                logger.warning("Meta-model is not fitted; falling back to default prediction")
                return self._fallback_prediction(X)
            meta_features_scaled = self.meta_scaler.transform(meta_features.values)
            meta_features_scaled_df = pd.DataFrame(meta_features_scaled, columns=expected_cols)
            logger.debug(f"Meta-features scaled shape: {meta_features_scaled_df.shape}")
            proba = self.meta_model.predict_proba(meta_features_scaled_df)[0, 1]
            prediction = 1 if proba >= 0.5 else 0
            confidence = abs(proba - 0.5) * 2
            logger.debug(f"Meta-model prediction: {proba} (class {prediction}) with confidence: {confidence}")
            logger.debug("Exiting predict_with_ensemble with success")
            return proba, confidence
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.debug("Exiting predict_with_ensemble with error; using fallback")
            return 0.5, 0

    def predict_proba(self, X):
        logger.debug("Entering predict_proba")
        probas = []
        for i in range(len(X)):
            sample = X.iloc[[i]] if hasattr(X, 'iloc') else X[i:i+1]
            logger.debug(f"Predicting for sample index: {i}")
            prob, _ = self.predict_with_ensemble(sample)
            probas.append(prob)
        logger.debug("Exiting predict_proba")
        return np.array(probas)

    def get_model_metrics(self):
        logger.debug("Entering get_model_metrics")
        logger.debug(f"Returning model metrics: {self.model_metrics}")
        return self.model_metrics

    def get_feature_importance(self):
        logger.debug("Entering get_feature_importance")
        try:
            if not self.feature_importance:
                logger.warning("No feature importance data available")
                return {}
            combined_importance = {}
            for model_name, importance in self.feature_importance.items():
                for feature, value in importance.items():
                    combined_importance.setdefault(feature, []).append(value)
            avg_importance = {feature: float(np.mean(values)) for feature, values in combined_importance.items()}
            total = sum(avg_importance.values())
            normalized_importance = {feature: float(value / total if total > 0 else 0) for feature, value in avg_importance.items()}
            sorted_importance = dict(sorted(normalized_importance.items(), key=lambda x: x[1], reverse=True))
            logger.debug(f"Computed feature importance: {sorted_importance}")
            return sorted_importance
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}

    def train_ensemble(self, data_frames, test_size=0.2, random_state=42, deep_learning=True, force_retrain=False):
        logger.debug("Entering train_ensemble")
        # If models are already loaded and force_retrain is False, skip retraining.
        if not force_retrain and self.models and self.meta_model is not None:
            logger.info("Using already trained ensemble models. To retrain, set force_retrain=True.")
            return {
                'success': True,
                'message': 'Using already trained models',
                'metrics': self.model_metrics,
                'lstm_trained': self.lstm_model is not None
            }
        if not data_frames:
            logger.error("No data provided for training")
            return {'success': False, 'message': 'No data provided'}
        try:
            all_data = pd.concat(data_frames, ignore_index=True)
            logger.debug(f"Combined data shape: {all_data.shape}")

            if 'Label' not in all_data.columns:
                logger.error("Label column not found in training data")
                return {'success': False, 'message': 'Label column not found'}

            exclude_cols = ['Label', 'Symbol', 'Timestamp', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            all_feature_cols = [col for col in all_data.columns if col not in exclude_cols]
            logger.debug(f"Feature columns: {all_feature_cols}")

            if not all_feature_cols:
                logger.error("No feature columns found")
                return {'success': False, 'message': 'No feature columns found'}

            if 'Timestamp' in all_data.columns:
                all_data = all_data.sort_values('Timestamp')
                logger.debug("Sorted data by Timestamp")

            train_idx = int(len(all_data) * (1 - test_size))
            train_data = all_data.iloc[:train_idx]
            test_data = all_data.iloc[train_idx:]
            logger.info(f"Train set: {len(train_data)} samples, Test set: {len(test_data)} samples")

            X_train = train_data[all_feature_cols]
            y_train = train_data['Label']
            X_test = test_data[all_feature_cols]
            y_test = test_data['Label']

            logger.debug(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            base_models_trained = self._train_base_models(X_train, y_train, X_test, y_test, all_feature_cols)
            if not base_models_trained:
                logger.error("Failed to train base models")
                return {'success': False, 'message': 'Base model training failed'}

            lstm_trained = False
            if deep_learning and TENSORFLOW_AVAILABLE:
                try:
                    logger.info("Training LSTM model...")
                    lstm_trained = self._train_lstm_model(all_data, all_feature_cols)
                    logger.info("LSTM model training successful" if lstm_trained else "LSTM model training failed")
                except Exception as e:
                    logger.error(f"Error training LSTM model: {e}")
                    lstm_trained = False

            meta_features_train, meta_features_test = self._create_meta_features(X_train, X_test)
            logger.debug(f"Meta-features train shape: {meta_features_train.shape}, test shape: {meta_features_test.shape}")
            meta_success = self._train_meta_learner(meta_features_train, y_train, meta_features_test, y_test)
            if not meta_success:
                logger.warning("Meta-learner training failed, will use base models only")

            ensemble_metrics = self._evaluate_ensemble(X_test, y_test)
            self.save_models()
            logger.info(f"Ensemble training completed with f2_score: {ensemble_metrics.get('f2_score', 0):.4f}")
            logger.debug("Exiting train_ensemble with success")
            return {
                'success': True,
                'message': 'Ensemble training completed successfully',
                'metrics': ensemble_metrics,
                'lstm_trained': lstm_trained
            }
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'message': f'Error: {str(e)}'}

    def _train_base_models(self, X_train, y_train, X_test, y_test, all_feature_cols):
        logger.debug("Entering _train_base_models")
        try:
            logger.info("Training base models...")
            # Define specialized feature subsets.
            tech_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
                'sma', 'ema', 'macd', 'rsi', 'bband', 'stoch', 'adx', 'atr', 'obv', 'cci', 'willr'
            ])]
            vol_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
                'volatility', 'garch', 'atr', 'range', 'std', 'bband'
            ])]
            mom_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
                'momentum', 'rsi', 'macd', 'roc', 'willr', 'stoch', 'trix', 'ao'
            ])]
            price_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
                'close', 'price', 'open', 'high', 'low', 'range', 'lag', 'change'
            ])]
            vol_ind_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
                'volume', 'vwap', 'obv', 'mfi', 'ad', 'cmf', 'vpt'
            ])]
            sent_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
                'sentiment', 'social', 'news', 'option', 'call', 'put', 'unusual'
            ])]
            regime_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
                'regime', 'trend', 'bull', 'bear', 'volatility', 'momentum'
            ])]
            order_features = [col for col in all_feature_cols if any(x in col.lower() for x in [
                'flow', 'imbalance', 'delta', 'vwap', 'dark', 'pressure'
            ])]

            feature_subsets = {
                'all_features': all_feature_cols,
                'technical': tech_features,
                'volatility': vol_features,
                'momentum': mom_features,
                'price_action': price_features,
                'volume': vol_ind_features,
                'sentiment': sent_features,
                'market_regime': regime_features,
                'order_flow': order_features
            }
            # Keep only subsets with at least 5 features.
            feature_subsets = {k: v for k, v in feature_subsets.items() if len(v) >= 5}
            if len(feature_subsets) < 3:
                logger.warning("Not enough specialized feature subsets, using random subsets")
                feature_subsets = {
                    'all_features': all_feature_cols,
                    'subset1': np.random.choice(all_feature_cols, min(50, len(all_feature_cols)), replace=False).tolist(),
                    'subset2': np.random.choice(all_feature_cols, min(50, len(all_feature_cols)), replace=False).tolist()
                }
            self.feature_subsets = feature_subsets.copy()
            logger.debug(f"Defined feature subsets: {self.feature_subsets}")

            model_configs = [
                {'name': 'xgb_all', 'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), 'subset': 'all_features'},
                {'name': 'lgb_all', 'model': LGBMClassifier(random_state=42), 'subset': 'all_features'},
                {'name': 'rf_all', 'model': RandomForestClassifier(n_estimators=100, random_state=42), 'subset': 'all_features'},
                {'name': 'gb_tech', 'model': GradientBoostingClassifier(random_state=42), 'subset': 'technical'},
                {'name': 'xgb_vol', 'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), 'subset': 'volatility'},
                {'name': 'lgb_mom', 'model': LGBMClassifier(random_state=42), 'subset': 'momentum'},
                {'name': 'rf_price', 'model': RandomForestClassifier(n_estimators=100, random_state=42), 'subset': 'price_action'},
                {'name': 'xgb_flow', 'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), 'subset': 'order_flow'}
            ]
            model_configs = [cfg for cfg in model_configs if cfg['subset'] in feature_subsets]

            for cfg in model_configs:
                subset_name = cfg['subset']
                model_name = cfg['name']
                model = cfg['model']
                features = feature_subsets[subset_name]
                if not features:
                    logger.warning(f"No features for subset {subset_name}, skipping model {model_name}")
                    continue
                logger.info(f"Training model '{model_name}' with {len(features)} features from subset '{subset_name}'")
                X_train_subset = X_train[features]
                X_test_subset = X_test[features]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_subset)
                X_test_scaled = scaler.transform(X_test_subset)

                model.fit(X_train_scaled, y_train)
                logger.debug(f"Model '{model_name}' trained; X_train_scaled shape: {X_train_scaled.shape}")

                self.models[model_name] = model
                self.scalers[model_name] = scaler
                self.feature_subsets[model_name] = features

                y_pred = model.predict(X_test_scaled)
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'f2_score': fbeta_score(y_test, y_pred, beta=2),
                    'auc': roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]) if hasattr(model, 'predict_proba') else 0.5
                }
                self.model_metrics[model_name] = metrics
                logger.info(f"Model '{model_name}' metrics: {metrics}")

                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(zip(features, model.feature_importances_))
            logger.debug("Exiting _train_base_models with success")
            return True
        except Exception as e:
            logger.error(f"Error training base models: {e}")
            logger.debug("Exiting _train_base_models with error")
            return False

    def _create_meta_features(self, X_train, X_test):
        logger.debug("Entering _create_meta_features")
        try:
            logger.info("Creating meta-features from base models...")
            meta_features_train = pd.DataFrame()
            meta_features_test = pd.DataFrame()
            for model_name, model in self.models.items():
                features = self.feature_subsets.get(model_name)
                if features is None:
                    logger.warning(f"Feature subset for model '{model_name}' not found; skipping.")
                    continue
                missing_train = [f for f in features if f not in X_train.columns]
                missing_test = [f for f in features if f not in X_test.columns]
                if missing_train or missing_test:
                    logger.warning(f"Model '{model_name}' missing features. Train missing: {missing_train}, Test missing: {missing_test}; skipping.")
                    continue
                scaler = self.scalers[model_name]
                X_train_scaled = scaler.transform(X_train[features])
                X_test_scaled = scaler.transform(X_test[features])
                if hasattr(model, 'predict_proba'):
                    train_preds = model.predict_proba(X_train_scaled)[:, 1]
                    test_preds = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    train_preds = model.predict(X_train_scaled)
                    test_preds = model.predict(X_test_scaled)
                meta_features_train[f"{model_name}_prob"] = train_preds
                meta_features_test[f"{model_name}_prob"] = test_preds
                logger.debug(f"Model '{model_name}': added meta predictions; train preds shape: {train_preds.shape}, test preds shape: {test_preds.shape}")
            logger.debug(f"Meta-features train before reindex: shape {meta_features_train.shape}, columns: {meta_features_train.columns.tolist()}")
            if self.meta_feature_names is None:
                expected_cols = meta_features_train.columns.tolist()
                logger.debug("Meta-feature names not preset; using dynamic columns: %s", expected_cols)
            else:
                expected_cols = [str(col) for col in self.meta_feature_names]
                logger.debug("Using stored meta_feature_names: %s", expected_cols)
            meta_features_train = meta_features_train.reindex(columns=expected_cols, fill_value=0.5)
            meta_features_test = meta_features_test.reindex(columns=expected_cols, fill_value=0.5)
            logger.info(f"Created meta-features: train shape {meta_features_train.shape}, test shape {meta_features_test.shape}")
            logger.debug("Exiting _create_meta_features with success")
            return meta_features_train, meta_features_test
        except Exception as e:
            logger.error(f"Error creating meta-features: {e}")
            logger.debug("Exiting _create_meta_features with error")
            return pd.DataFrame(), pd.DataFrame()

    def _train_meta_learner(self, meta_features_train, y_train, meta_features_test, y_test):
        logger.debug("Entering _train_meta_learner")
        try:
            if meta_features_train.empty or meta_features_test.empty:
                logger.warning("Empty meta-features; cannot train meta-learner")
                return False
            logger.info("Training meta-learner...")
            # Ensure both train and test meta-features have the same columns
            all_columns = set(meta_features_train.columns) | set(meta_features_test.columns)
            for col in all_columns:
                if col not in meta_features_train.columns:
                    meta_features_train[col] = 0.0
                if col not in meta_features_test.columns:
                    meta_features_test[col] = 0.0
            logger.debug(f"Meta-features train after alignment: shape {meta_features_train.shape}")

            self.meta_scaler = StandardScaler()
            meta_train_scaled = self.meta_scaler.fit_transform(meta_features_train.values)
            meta_test_scaled = self.meta_scaler.transform(meta_features_test.values)
            logger.debug(f"Meta train scaled shape: {meta_train_scaled.shape}, test scaled shape: {meta_test_scaled.shape}")

            from xgboost import XGBClassifier
            self.meta_model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
            self.meta_feature_names = meta_features_train.columns.tolist()
            logger.debug(f"Meta-feature names set to: {self.meta_feature_names}")

            self.meta_model.fit(meta_train_scaled, y_train)
            logger.debug("Meta-model training completed.")

            y_pred = self.meta_model.predict(meta_test_scaled)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'f2_score': fbeta_score(y_test, y_pred, beta=2),
                'auc': roc_auc_score(y_test, self.meta_model.predict_proba(meta_test_scaled)[:, 1])
            }
            self.model_metrics['meta_learner'] = metrics
            logger.info(f"Meta-learner trained with metrics: {metrics}")
            logger.debug("Exiting _train_meta_learner with success")
            return True
        except Exception as e:
            logger.error(f"Error training meta-learner: {e}")
            logger.debug("Exiting _train_meta_learner with error")
            return False

    def _evaluate_ensemble(self, X_test, y_test):
        logger.debug("Entering _evaluate_ensemble")
        try:
            logger.info("Evaluating ensemble performance...")
            y_preds = []
            probas = []
            for i in range(len(X_test)):
                sample = X_test.iloc[[i]] if hasattr(X_test, 'iloc') else X_test[i:i+1]
                logger.debug(f"Evaluating sample index: {i}")
                prob, _ = self.predict_with_ensemble(sample)
                pred = 1 if prob > 0.5 else 0
                y_preds.append(pred)
                probas.append(prob)
            y_preds = np.array(y_preds)
            probas = np.array(probas)
            metrics = {
                'accuracy': accuracy_score(y_test, y_preds),
                'f1_score': f1_score(y_test, y_preds),
                'f2_score': fbeta_score(y_test, y_preds, beta=2),
                'auc': roc_auc_score(y_test, probas)
            }
            self.model_metrics['ensemble'] = metrics
            logger.info(f"Ensemble evaluation metrics: {metrics}")
            logger.debug("Exiting _evaluate_ensemble with success")
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {e}")
            return {'error': str(e)}

    def _train_lstm_model(self, all_data, all_feature_cols):
        logger.debug("Entering _train_lstm_model")
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, skipping LSTM training")
            return False
        try:
            logger.info("Training LSTM model...")
            X_sequences, y = self._prepare_sequence_data(all_data, all_feature_cols)
            if X_sequences is None or y is None:
                logger.warning("Could not prepare sequence data for LSTM")
                return False
            train_size = int(len(X_sequences) * 0.8)
            X_train, X_test = X_sequences[:train_size], X_sequences[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            logger.debug(f"LSTM training: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
            n_features = X_train.shape[2]
            model = Sequential()
            model.add(LSTM(units=64, input_shape=(self.lstm_sequence_length, n_features), return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(LSTM(units=32, return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            results = model.evaluate(X_test, y_test)
            lstm_path = os.path.join(self.model_dir, "lstm_model.keras")
            model.save(lstm_path)
            self.model_metrics['lstm'] = {
                'loss': float(results[0]),
                'accuracy': float(results[1])
            }
            logger.info(f"LSTM model trained with accuracy: {results[1]:.4f}")
            self.lstm_model = model
            logger.debug("Exiting _train_lstm_model with success")
            return True
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _prepare_sequence_data(self, df, feature_cols, target_col='Label'):
        logger.debug("Entering _prepare_sequence_data")
        try:
            if 'Timestamp' in df.columns:
                df = df.sort_values('Timestamp').reset_index(drop=True)
            X = df[feature_cols].values
            y = df[target_col].values
            X_sequences = []
            y_target = []
            for i in range(len(X) - self.lstm_sequence_length):
                X_sequences.append(X[i:(i + self.lstm_sequence_length)])
                y_target.append(y[i + self.lstm_sequence_length])
            logger.debug("Exiting _prepare_sequence_data with success")
            return np.array(X_sequences), np.array(y_target)
        except Exception as e:
            logger.error(f"Error preparing sequence data: {e}")
            return None, None

    def _prepare_lstm_input(self, X):
        logger.debug("Entering _prepare_lstm_input")
        try:
            all_features = set()
            for features in self.feature_subsets.values():
                if isinstance(features, list):
                    all_features.update(features)
            missing_features = [f for f in all_features if f not in X.columns]
            if missing_features:
                logger.warning(f"Missing features for LSTM input: {missing_features}")
                return None
            X_subset = X[list(all_features)]
            if len(X_subset) == 1:
                sequence = np.repeat(X_subset.values, self.lstm_sequence_length, axis=0)
                sequence = sequence.reshape(1, self.lstm_sequence_length, -1)
                logger.debug("Exiting _prepare_lstm_input with single sample")
                return sequence
            else:
                sequences = []
                for i in range(len(X_subset) - self.lstm_sequence_length + 1):
                    seq = X_subset.iloc[i:i+self.lstm_sequence_length].values
                    sequences.append(seq)
                logger.debug("Exiting _prepare_lstm_input with multiple samples")
                return np.array(sequences)
        except Exception as e:
            logger.error(f"Error preparing LSTM input: {e}")
            return None

    # Alias method for compatibility.
    def train_model(self, data_frames, test_size=0.2, random_state=42, deep_learning=True, force_retrain=False):
        return self.train_ensemble(data_frames, test_size=test_size, random_state=random_state, deep_learning=deep_learning, force_retrain=force_retrain)


# Demo usage if run as main
if __name__ == "__main__":
    import yfinance as yf
    from advanced_feature_engineering import AdvancedFeatureEngineering

    ticker = "AAPL"
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, period="1y")
    data["Symbol"] = ticker
    data.reset_index(inplace=True)
    data['Label'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    fe = AdvancedFeatureEngineering()
    enhanced_data = fe.add_all_features(data)
    enhanced_data['Label'] = (enhanced_data['Close'].shift(-1) > enhanced_data['Close']).astype(int)
    enhanced_data = enhanced_data.dropna()
    logger.info(f"Training data shape: {enhanced_data.shape}")
    logger.info(f"Label column distribution: {enhanced_data['Label'].value_counts().to_dict()}")
    logger.info(f"Training data columns: {enhanced_data.columns.tolist()}")

    ensemble = MLTradingPredictor(model_dir="./ensemble_models")
    print("Training ensemble...")
    # Use force_retrain=True if you want to retrain even when models are already loaded.
    result = ensemble.train_model([enhanced_data], deep_learning=TENSORFLOW_AVAILABLE, force_retrain=False)
    print(f"Training result: {result}")

    importance = ensemble.get_feature_importance()
    print("\nTop 10 features by importance:")
    for feature, value in list(importance.items())[:10]:
        print(f"{feature}: {value:.4f}")

    last_data = enhanced_data.tail(1)
    prediction, confidence = ensemble.predict_with_ensemble(last_data)
    print(f"\nPrediction for {ticker}: {'INCREASE' if prediction > 0.5 else 'DECREASE'}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Probability: {ensemble.predict_proba(last_data)[0]:.4f}")
    print("Saving models...")
    ensemble.save_models()
    print("Models saved successfully")
    print("Loading models...")
    ensemble.load_models()
    print("Models loaded successfully")
    print("Predicting with loaded models...")
    prediction, confidence = ensemble.predict_with_ensemble(last_data)
    print(f"\nPrediction for {ticker}: {'INCREASE' if prediction > 0.5 else 'DECREASE'}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Probability: {ensemble.predict_proba(last_data)[0]:.4f}")
    print("Getting model metrics...")
    metrics = ensemble.get_model_metrics()
    print("Model metrics:")
    for model_name, model_metrics in metrics.items():
        print(f"{model_name}: {model_metrics}")
    print("Getting feature importance...")
    importance = ensemble.get_feature_importance()
    print("\nTop 10 features by importance:")
    for feature, value in list(importance.items())[:10]:
        print(f"{feature}: {value:.4f}")
    print("Done")
