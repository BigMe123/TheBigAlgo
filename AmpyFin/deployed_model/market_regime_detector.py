# market_regime_detector.py
import numpy as np
import pandas as pd
import os
import logging
import joblib
import time
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal

# Additional imports for Bayesian HMM
from pomegranate.hmm import BayesianHMM
from pomegranate.distributions import Normal as NormalDistribution
from pomegranate.distributions import MultivariateGaussianDistribution

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Detects market regimes using various models including:
    - Hidden Markov Models (HMM)
    - Bayesian Hidden Markov Models (BHMM) with Dirichlet priors
    - Particle Filters (Sequential Monte Carlo)
    - Gaussian Mixture Models (GMM)
    - K-Means clustering
    
    Enables adaptive strategy switching based on the current market regime.
    """
    
    def __init__(self, model_dir="./regime_models", n_regimes=3):
        """
        Initialize the MarketRegimeDetector with specified parameters.
        
        Args:
            model_dir: Directory to save/load model files
            n_regimes: Number of regimes to detect (set to 0 for dynamic regime selection with BHMM)
        """
        
        self.model_dir = model_dir
        self.n_regimes = n_regimes
        
        # HMM model for regime detection
        self.hmm_model = None
        
        # Bayesian HMM model using pomegranate
        self.bhmm_model = None
        
        # Particle filter for regime detection
        self.particle_filter = None
        
        # GMM model for clustering
        self.gmm_model = None
        
        # K-Means model for simpler clustering
        self.kmeans_model = None
        
        # Feature scaler
        self.scaler = None
        
        # Feature selection
        self.selected_features = None
        
        # Regime characteristics
        self.regime_characteristics = None
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def fit(self, data_frames, method="hmm"):
        """
        Train the regime detection model using the specified method.
        
        Args:
            data_frames: List of DataFrames containing market data
            method: Detection method to use ('hmm', 'bhmm', 'particle_filter', 'gmm', or 'kmeans')
            
        Returns:
            dict: Training results and statistics
        """
        
        try:
            # Combine all dataframes
            data = pd.concat(data_frames, ignore_index=True)
            
            # Sort by timestamp if available
            if 'Timestamp' in data.columns:
                data = data.sort_values('Timestamp')
            
            # Select relevant features for regime detection
            self.selected_features = self._select_regime_features(data)
            
            if not self.selected_features:
                logger.warning("No suitable features found for regime detection")
                return {"success": False, "message": "No suitable features found"}
            
            # Extract feature matrix
            X = data[self.selected_features].values
            
            # Standardize features
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train selected model
            if method == "hmm":
                success = self._train_hmm(X_scaled)
            elif method == "bhmm":  # New Bayesian HMM option
                success = self._train_bhmm(X_scaled)
            elif method == "particle_filter":  # New Particle Filter option
                success = self._train_particle_filter(X_scaled)
            elif method == "gmm":
                success = self._train_gmm(X_scaled)
            elif method == "kmeans":
                success = self._train_kmeans(X_scaled)
            else:
                logger.warning(f"Unknown method: {method}, using HMM")
                success = self._train_hmm(X_scaled)
            
            if not success:
                logger.error("Failed to train regime detection model")
                return {"success": False, "message": "Model training failed"}
            
            # Predict regimes for all data
            regimes = self.predict_regimes(data)
            
            # Extract regime characteristics
            self._extract_regime_characteristics(data, regimes)
            
            # Save models
            self.save_models()
            
            return {
                "success": True,
                "message": f"Successfully trained {method.upper()} model",
                "n_regimes": self.n_regimes,
                "regime_counts": np.bincount(regimes),
                "features_used": self.selected_features
            }
        
        except Exception as e:
            logger.error(f"Error training regime detection model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def _select_regime_features(self, data):
        """
        Select most relevant features for regime detection.
        
        Args:
            data: DataFrame with training data
            
        Returns:
            list: Selected feature names
        """
        # Exclude non-feature columns
        exclude_cols = ['Symbol', 'Timestamp', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Label']
        candidate_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Look for specific types of features relevant for regime detection
        regime_features = []
        
        # Volatility features
        vol_features = [col for col in candidate_cols if any(x in col.lower() for x in [
            'volatility', 'atr', 'garch', 'std', 'vix', 'range'
        ])]
        regime_features.extend(vol_features[:2])  # Limit to top 2
        
        # Trend features
        trend_features = [col for col in candidate_cols if any(x in col.lower() for x in [
            'trend', 'macd', 'adx', 'slope', 'direction', 'momentum'
        ])]
        regime_features.extend(trend_features[:2])  # Limit to top 2
        
        # Mean reversion features
        mean_rev_features = [col for col in candidate_cols if any(x in col.lower() for x in [
            'zscore', 'mean', 'reversion', 'rsi', 'stoch'
        ])]
        regime_features.extend(mean_rev_features[:2])  # Limit to top 2
        
        # Volume features
        vol_ind_features = [col for col in candidate_cols if any(x in col.lower() for x in [
            'volume', 'obv', 'flow', 'mfi'
        ])]
        regime_features.extend(vol_ind_features[:1])  # Limit to top 1
        
        # Market breadth/sentiment features
        sentiment_features = [col for col in candidate_cols if any(x in col.lower() for x in [
            'sentiment', 'breadth', 'advance', 'decline'
        ])]
        regime_features.extend(sentiment_features[:1])  # Limit to top 1
        
        # If we don't have enough specific features, add some general ones
        if len(regime_features) < 4:
            # Add return and correlation features
            for col in candidate_cols:
                if 'return' in col.lower() or 'correlation' in col.lower() or 'change' in col.lower():
                    regime_features.append(col)
                    if len(regime_features) >= 6:
                        break
        
        # If still not enough, add price relative to moving averages
        if len(regime_features) < 4:
            for col in candidate_cols:
                if 'sma' in col.lower() or 'ema' in col.lower() or 'price' in col.lower():
                    regime_features.append(col)
                    if len(regime_features) >= 6:
                        break
        
        # If we still don't have enough features, use all numeric features (up to a limit)
        if len(regime_features) < 4:
            numeric_cols = data.select_dtypes(include=['number']).columns
            additional_cols = [col for col in numeric_cols if col not in exclude_cols and col not in regime_features]
            regime_features.extend(additional_cols[:8 - len(regime_features)])
        
        # Return unique features
        return list(set(regime_features))
    
    def _train_hmm(self, X_scaled):
        """
        Train a Hidden Markov Model for regime detection.
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            bool: Success or failure
        """
        
        try:
            logger.info(f"Training HMM with {self.n_regimes} states...")
            
            # Initialize HMM model
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            
            # Fit model
            self.hmm_model.fit(X_scaled)
            
            logger.info(f"HMM training completed with score: {self.hmm_model.score(X_scaled):.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training HMM model: {e}")
            return False
    
    def _train_bhmm(self, X_scaled):
        """
        Train a Bayesian Hidden Markov Model with Dirichlet priors for transition probabilities.
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            bool: Success or failure
        """
        try:
            # Determine optimal number of regimes if specified as 0
            if self.n_regimes <= 0:
                logger.info("Performing dynamic regime selection...")
                best_n_regimes = self._determine_optimal_regimes(X_scaled)
                logger.info(f"Optimal number of regimes determined to be {best_n_regimes}")
                self.n_regimes = best_n_regimes
            else:
                logger.info(f"Training Bayesian HMM with {self.n_regimes} states...")
            
            # Initialize distributions for each state
            n_features = X_scaled.shape[1]
            
            if n_features == 1:
                # For one-dimensional data
                distributions = [NormalDistribution() for _ in range(self.n_regimes)]
            else:
                # For multi-dimensional data
                distributions = [MultivariateGaussianDistribution() for _ in range(self.n_regimes)]
            
            # Initialize Bayesian HMM with Dirichlet priors
            # The edge_inertia parameter controls the Dirichlet prior strength for transition probabilities
            # Higher values make the model more resistant to changing transition probabilities
            self.bhmm_model = BayesianHMM(
                distributions=distributions,
                edge_inertia=0.5,  # Dirichlet prior for transition probabilities
                distribution_inertia=0.5  # Dirichlet prior for emission probabilities
            )
            
            # Fit model
            self.bhmm_model.fit(X_scaled)
            
            logger.info("Bayesian HMM training completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training Bayesian HMM model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _determine_optimal_regimes(self, X_scaled, max_regimes=6):
        """
        Determine the optimal number of regimes using BIC criterion.
        This allows for dynamic regime selection rather than fixing states.
        
        Args:
            X_scaled: Scaled feature matrix
            max_regimes: Maximum number of regimes to consider
            
        Returns:
            int: Optimal number of regimes
        """
        try:
            # Start with 2 regimes as a minimum
            min_regimes = 2
            bic_scores = []
            
            logger.info(f"Determining optimal number of regimes (2-{max_regimes})...")
            
            for n in range(min_regimes, max_regimes + 1):
                # Create and train a model with n regimes
                n_features = X_scaled.shape[1]
                if n_features == 1:
                    # For one-dimensional data
                    distributions = [NormalDistribution() for _ in range(n)]
                else:
                    # For multi-dimensional data
                    distributions = [MultivariateGaussianDistribution() for _ in range(n)]
                
                model = BayesianHMM(
                    distributions=distributions,
                    edge_inertia=0.5,
                    distribution_inertia=0.5
                )
                
                # Fit model
                model.fit(X_scaled)
                
                # Calculate BIC score
                log_likelihood = model.log_probability(X_scaled).sum()
                n_params = n * n  # Transition matrix
                n_params += n * (n_features + n_features * (n_features + 1) / 2)  # Emission distributions
                bic = -2 * log_likelihood + n_params * np.log(len(X_scaled))
                
                bic_scores.append(bic)
                logger.info(f"BIC score for {n} regimes: {bic}")
            
            # Find the model with the lowest BIC score
            optimal_n_regimes = np.argmin(bic_scores) + min_regimes
            
            return optimal_n_regimes
            
        except Exception as e:
            logger.error(f"Error determining optimal regimes: {e}")
            # Default to 3 regimes if there's an error
            return 3
    
    def _initialize_particle_filter(self, n_particles=1000):
        """
        Initialize the particle filter for regime detection.
        
        Args:
            n_particles: Number of particles to use (1000+ recommended)
            
        Returns:
            dict: Particle filter state
        """
        try:
            logger.info(f"Initializing particle filter with {n_particles} particles...")
            
            # Create particle filter state
            particle_filter = {
                'n_particles': n_particles,
                'n_regimes': self.n_regimes,
                'particles': np.random.randint(0, self.n_regimes, n_particles),  # Initial regime assignments
                'weights': np.ones(n_particles) / n_particles,  # Uniform initial weights
                'transition_matrix': self._estimate_transition_matrix(),  # Estimated regime transition probabilities
                'emission_models': self._estimate_emission_models()  # Emission probability models for each regime
            }
            
            return particle_filter
            
        except Exception as e:
            logger.error(f"Error initializing particle filter: {e}")
            return None
    
    def _estimate_transition_matrix(self):
        """
        Estimate regime transition probabilities from existing model if available.
        
        Returns:
            numpy.ndarray: Transition probability matrix
        """
        try:
            # If we have an HMM model, use its transition matrix
            if self.hmm_model is not None:
                return self.hmm_model.transmat_
            
            # If we have a Bayesian HMM model, use its transition matrix
            if hasattr(self, 'bhmm_model') and self.bhmm_model is not None:
                return self.bhmm_model.dense_transition_matrix()
            
            # Otherwise, create a reasonable default with high self-transition probability
            transition_matrix = np.ones((self.n_regimes, self.n_regimes)) * 0.1
            np.fill_diagonal(transition_matrix, 0.7)  # High probability of staying in the same regime
            
            # Normalize rows to sum to 1
            row_sums = transition_matrix.sum(axis=1)
            return transition_matrix / row_sums[:, np.newaxis]
            
        except Exception as e:
            logger.error(f"Error estimating transition matrix: {e}")
            # Return uniform transition probabilities
            return np.ones((self.n_regimes, self.n_regimes)) / self.n_regimes
    
    def _estimate_emission_models(self):
        """
        Estimate emission probability models for each regime.
        
        Returns:
            list: List of emission models (multivariate normal distributions) for each regime
        """
        try:
            emission_models = []
            
            # If we have trained an HMM model, use its parameters
            if self.hmm_model is not None:
                for i in range(self.n_regimes):
                    mean = self.hmm_model.means_[i]
                    covariance = self.hmm_model.covars_[i]
                    emission_models.append((mean, covariance))
                
                return emission_models
            
            # If we have a Bayesian HMM model, use its parameters
            if hasattr(self, 'bhmm_model') and self.bhmm_model is not None:
                for i in range(self.n_regimes):
                    distribution = self.bhmm_model.distributions[i]
                    if hasattr(distribution, 'parameters'):
                        mean = distribution.parameters[0]
                        covariance = distribution.parameters[1]
                        emission_models.append((mean, covariance))
                
                if emission_models:
                    return emission_models
            
            # Otherwise, create default emission models based on clustering the data
            from sklearn.cluster import KMeans
            
            # Get data from scaler if available
            if self.scaler is not None and hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
                mean = self.scaler.mean_
                variance = self.scaler.scale_ ** 2
                
                # Create basic emission models
                for i in range(self.n_regimes):
                    # Create slightly different means for each regime
                    regime_mean = mean + (i - self.n_regimes // 2) * np.sqrt(variance) * 0.5
                    # Use the variance as the covariance matrix
                    if len(mean) == 1:
                        regime_covariance = variance
                    else:
                        regime_covariance = np.diag(variance)
                    
                    emission_models.append((regime_mean, regime_covariance))
                
                return emission_models
            
            # If no data is available, return None and handle this case later
            return None
            
        except Exception as e:
            logger.error(f"Error estimating emission models: {e}")
            return None
    
    def _update_particle_filter(self, X_scaled, particle_filter):
        """
        Update particle filter with new observation.
        
        Args:
            X_scaled: Scaled observation (single time point)
            particle_filter: Current particle filter state
            
        Returns:
            dict: Updated particle filter state
        """
        try:
            n_particles = particle_filter['n_particles']
            n_regimes = particle_filter['n_regimes']
            particles = particle_filter['particles']
            weights = particle_filter['weights']
            transition_matrix = particle_filter['transition_matrix']
            emission_models = particle_filter['emission_models']
            
            # Propagate particles through transition model
            new_particles = np.zeros(n_particles, dtype=int)
            for i in range(n_particles):
                current_regime = particles[i]
                # Sample new regime from transition probabilities
                new_particles[i] = np.random.choice(n_regimes, p=transition_matrix[current_regime])
            
            # Update weights based on emission probabilities (market signal likelihoods)
            new_weights = np.zeros(n_particles)
            for i in range(n_particles):
                regime = new_particles[i]
                mean, covariance = emission_models[regime]
                
                # Calculate emission probability (likelihood of the observation given the regime)
                try:
                    # For multivariate case
                    emission_prob = multivariate_normal.pdf(X_scaled, mean=mean, cov=covariance)
                except:
                    # For scalar case or if there's an error in multivariate calculation
                    # Fall back to manual calculation of normal PDF
                    emission_prob = 1.0
                    for j in range(len(X_scaled)):
                        try:
                            # For diagonal covariance
                            var = covariance[j, j] if isinstance(covariance, np.ndarray) and covariance.ndim > 1 else covariance[j]
                            emission_prob *= np.exp(-0.5 * ((X_scaled[j] - mean[j]) ** 2) / var)
                            emission_prob /= np.sqrt(2 * np.pi * var)
                        except:
                            # If all else fails, use a default value
                            emission_prob *= 0.1
                
                # Update weight
                new_weights[i] = weights[i] * emission_prob
            
            # Normalize weights
            weight_sum = np.sum(new_weights)
            if weight_sum > 0:
                new_weights = new_weights / weight_sum
            else:
                # If all weights are zero, reset to uniform weights
                new_weights = np.ones(n_particles) / n_particles
            
            # Check for weight degeneracy
            n_eff = 1.0 / np.sum(new_weights ** 2)
            
            # Resample if effective sample size is too small (weight degeneracy detected)
            if n_eff < n_particles / 2:
                indices = np.random.choice(n_particles, n_particles, p=new_weights)
                new_particles = new_particles[indices]
                new_weights = np.ones(n_particles) / n_particles
            
            # Update particle filter state
            particle_filter['particles'] = new_particles
            particle_filter['weights'] = new_weights
            
            return particle_filter
            
        except Exception as e:
            logger.error(f"Error updating particle filter: {e}")
            return particle_filter
    
    def _train_particle_filter(self, X_scaled):
        """
        Train a particle filter (Sequential Monte Carlo) for regime detection.
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            bool: Success or failure
        """
        try:
            logger.info("Training particle filter for regime detection...")
            
            # Initialize particle filter with 1000+ particles for accuracy
            self.particle_filter = self._initialize_particle_filter(n_particles=1000)
            
            if self.particle_filter is None:
                logger.error("Failed to initialize particle filter")
                return False
            
            # If emission models are not available, we need to estimate them from data
            if self.particle_filter['emission_models'] is None:
                logger.info("Estimating emission models from data...")
                
                # Use K-Means to estimate initial regime parameters
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
                cluster_labels = kmeans.fit_predict(X_scaled)
                
                # Calculate means and covariances for each cluster
                emission_models = []
                for i in range(self.n_regimes):
                    cluster_data = X_scaled[cluster_labels == i]
                    if len(cluster_data) > 0:
                        mean = np.mean(cluster_data, axis=0)
                        # Add small regularization to avoid singular matrices
                        cov = np.cov(cluster_data, rowvar=False) + 1e-6 * np.eye(X_scaled.shape[1])
                        if cov.ndim == 0:  # Handle 1D case
                            cov = np.array([[cov]])
                    else:
                        # Fallback if a cluster is empty
                        mean = np.zeros(X_scaled.shape[1])
                        cov = np.eye(X_scaled.shape[1])
                    
                    emission_models.append((mean, cov))
                
                self.particle_filter['emission_models'] = emission_models
            
            # Update particle filter for each observation
            if len(X_scaled) > 0:
                logger.info(f"Updating particle filter with {len(X_scaled)} observations...")
                start_time = time.time()
                
                # Process observations sequentially
                for i, x in enumerate(X_scaled):
                    # Log progress periodically
                    if i % 100 == 0 and i > 0:
                        elapsed = time.time() - start_time
                        remaining = (elapsed / i) * (len(X_scaled) - i)
                        logger.info(f"Processed {i}/{len(X_scaled)} observations... " +
                                   f"(Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s)")
                    
                    # Update particle filter with current observation
                    self.particle_filter = self._update_particle_filter(x, self.particle_filter)
                
                end_time = time.time()
                logger.info(f"Particle filter training completed in {end_time - start_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training particle filter: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _predict_regime_particle_filter(self, X_scaled):
        """
        Predict regime using particle filter (Sequential Monte Carlo).
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            numpy.ndarray: Array of regime labels
        """
        try:
            if self.particle_filter is None:
                logger.warning("Particle filter not initialized")
                return np.zeros(len(X_scaled))
            
            # Make a copy of the particle filter to avoid modifying the original
            pf = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self.particle_filter.items()}
            
            # Array to store regime predictions
            regimes = np.zeros(len(X_scaled), dtype=int)
            
            # Process each observation
            for i, x in enumerate(X_scaled):
                # Update particle filter
                pf = self._update_particle_filter(x, pf)
                
                # Predict regime as the most common regime among particles (weighted)
                regime_counts = np.bincount(pf['particles'], weights=pf['weights'], minlength=self.n_regimes)
                regimes[i] = np.argmax(regime_counts)
            
            return regimes
            
        except Exception as e:
            logger.error(f"Error predicting regimes with particle filter: {e}")
            return np.zeros(len(X_scaled))
    
    def _train_gmm(self, X_scaled):
        """
        Train Gaussian Mixture Model for regime clustering.
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            bool: Success or failure
        """
        
        try:
            logger.info(f"Training GMM with {self.n_regimes} components...")
            
            # Initialize GMM model
            self.gmm_model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type="full",
                max_iter=100,
                random_state=42
            )
            
            # Fit model
            self.gmm_model.fit(X_scaled)
            
            logger.info("GMM training completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training GMM model: {e}")
            return False
    
    def _train_kmeans(self, X_scaled):
        """
        Train K-Means for simpler regime clustering.
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            bool: Success or failure
        """
        try:
            logger.info(f"Training K-Means with {self.n_regimes} clusters...")
            
            # Initialize K-Means model
            self.kmeans_model = KMeans(
                n_clusters=self.n_regimes,
                random_state=42,
                n_init=10
            )
            
            # Fit model
            self.kmeans_model.fit(X_scaled)
            
            logger.info("K-Means training completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training K-Means model: {e}")
            return False
    
    def predict_regimes(self, data):
        """
        Predict market regimes for the given data.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            numpy.ndarray: Array of regime labels
        """
        try:
            # Check if models are available
            if not (self.hmm_model or hasattr(self, 'bhmm_model') and self.bhmm_model or 
                    hasattr(self, 'particle_filter') and self.particle_filter or 
                    self.gmm_model or self.kmeans_model):
                logger.warning("No models available for regime prediction")
                # Return default regime (0)
                return np.zeros(len(data))
            
            # Extract features
            if not all(feature in data.columns for feature in self.selected_features):
                logger.warning("Not all required features are available in data")
                missing_features = [f for f in self.selected_features if f not in data.columns]
                logger.warning(f"Missing features: {missing_features}")
                return np.zeros(len(data))
            
            X = data[self.selected_features].values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict regimes using available model
            if hasattr(self, 'bhmm_model') and self.bhmm_model:
                # Use Bayesian HMM for prediction
                regimes = self.bhmm_model.predict(X_scaled)
            elif hasattr(self, 'particle_filter') and self.particle_filter:
                # Use Particle Filter for prediction
                regimes = self._predict_regime_particle_filter(X_scaled)
            elif self.hmm_model:
                # Use standard HMM for prediction
                regimes = self.hmm_model.predict(X_scaled)
            elif self.gmm_model:
                regimes = self.gmm_model.predict(X_scaled)
            elif self.kmeans_model:
                regimes = self.kmeans_model.predict(X_scaled)
            else:
                regimes = np.zeros(len(data))
            
            return regimes
            
        except Exception as e:
            logger.error(f"Error predicting regimes: {e}")
            return np.zeros(len(data))
    
    def _extract_regime_characteristics(self, data, regimes):
        """
        Extract and store characteristics of each regime.
        
        Args:
            data: DataFrame with market data
            regimes: Array of regime labels
        """
        try:
            # Create regime characteristics dictionary
            self.regime_characteristics = {}
            
            # Add regime column to data
            data_with_regimes = data.copy()
            data_with_regimes['Regime'] = regimes
            
            # Calculate characteristics for each regime
            for regime in range(self.n_regimes):
                regime_data = data_with_regimes[data_with_regimes['Regime'] == regime]
                
                if len(regime_data) == 0:
                    continue
                
                # Calculate basic statistics
                stats = {}
                
                # Return statistics
                if 'Close' in data.columns:
                    regime_returns = regime_data['Close'].pct_change().dropna()
                    stats['avg_return'] = regime_returns.mean()
                    stats['volatility'] = regime_returns.std()
                    stats['sharpe'] = stats['avg_return'] / stats['volatility'] if stats['volatility'] > 0 else 0
                    stats['max_drawdown'] = self._calculate_max_drawdown(regime_data['Close'])
                    stats['win_rate'] = (regime_returns > 0).mean()
                    stats['avg_win'] = regime_returns[regime_returns > 0].mean() if any(regime_returns > 0) else 0
                    stats['avg_loss'] = regime_returns[regime_returns < 0].mean() if any(regime_returns < 0) else 0
                
                # Trend strength
                if 'ADX' in data.columns:
                    stats['trend_strength'] = regime_data['ADX'].mean()
                
                # Volatility
                if any('Volatility' in col for col in data.columns):
                    vol_col = next(col for col in data.columns if 'Volatility' in col)
                    stats['volatility_indicator'] = regime_data[vol_col].mean()
                
                # Regime duration
                regime_runs = self._identify_regime_runs(regimes, regime)
                stats['avg_duration'] = np.mean(regime_runs) if regime_runs else 0
                stats['max_duration'] = np.max(regime_runs) if regime_runs else 0
                
                # Days in regime
                stats['days_in_regime'] = len(regime_data)
                stats['pct_in_regime'] = len(regime_data) / len(data)
                
                # Store characteristics
                self.regime_characteristics[regime] = stats
            
            logger.info("Regime characteristics extracted")
            
        except Exception as e:
            logger.error(f"Error extracting regime characteristics: {e}")
    
    def _calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown for a price series."""
        # Generate cumulative returns
        returns = prices.pct_change().dropna()
        cum_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.cummax()
        
        # Calculate drawdown
        drawdown = (cum_returns / running_max) - 1
        
        # Get maximum drawdown
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    def _identify_regime_runs(self, regimes, target_regime):
        """Identify consecutive runs of a particular regime."""
        in_run = False
        runs = []
        current_run = 0
        
        for r in regimes:
            if r == target_regime:
                in_run = True
                current_run += 1
            else:
                if in_run:
                    runs.append(current_run)
                    current_run = 0
                in_run = False
        
        # Don't forget the last run if it's still going
        if in_run:
            runs.append(current_run)
        
        return runs
    
    def save_models(self):
        """Save trained models to disk."""
        try:
            # Create directory if it doesn't exist
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            
            # Save HMM model
            if self.hmm_model:
                joblib.dump(self.hmm_model, os.path.join(self.model_dir, "hmm_model.joblib"))
            
            # Save Bayesian HMM model
            if hasattr(self, 'bhmm_model') and self.bhmm_model:
                joblib.dump(self.bhmm_model, os.path.join(self.model_dir, "bhmm_model.joblib"))
            
            # Save Particle Filter
            if hasattr(self, 'particle_filter') and self.particle_filter:
                joblib.dump(self.particle_filter, os.path.join(self.model_dir, "particle_filter.joblib"))
            
            # Save GMM model
            if self.gmm_model:
                joblib.dump(self.gmm_model, os.path.join(self.model_dir, "gmm_model.joblib"))
            
            # Save K-Means model
            if self.kmeans_model:
                joblib.dump(self.kmeans_model, os.path.join(self.model_dir, "kmeans_model.joblib"))
            
            # Save scaler
            if self.scaler:
                joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.joblib"))
            
            # Save feature names
            if self.selected_features:
                joblib.dump(self.selected_features, os.path.join(self.model_dir, "features.joblib"))
            
            # Save regime characteristics
            if self.regime_characteristics:
                joblib.dump(self.regime_characteristics, os.path.join(self.model_dir, "regime_characteristics.joblib"))
            
            # Save metadata
            metadata = {
                "n_regimes": self.n_regimes,
                "features": self.selected_features,
                "models": {
                    "hmm": self.hmm_model is not None,
                    "bhmm": hasattr(self, 'bhmm_model') and self.bhmm_model is not None,
                    "particle_filter": hasattr(self, 'particle_filter') and self.particle_filter is not None,
                    "gmm": self.gmm_model is not None,
                    "kmeans": self.kmeans_model is not None
                },
                "last_updated": datetime.now().isoformat()
            }
            
            joblib.dump(metadata, os.path.join(self.model_dir, "metadata.joblib"))
            
            logger.info(f"Models saved to {self.model_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self):
        """Load trained models from disk."""
        try:
            # Check if directory exists
            if not os.path.exists(self.model_dir):
                logger.warning(f"Model directory {self.model_dir} does not exist")
                return False
            
            # Load metadata
            metadata_path = os.path.join(self.model_dir, "metadata.joblib")
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.n_regimes = metadata.get("n_regimes", self.n_regimes)
                self.selected_features = metadata.get("features", [])
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, "scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            # Load HMM model
            hmm_path = os.path.join(self.model_dir, "hmm_model.joblib")
            if os.path.exists(hmm_path):
                self.hmm_model = joblib.load(hmm_path)
            
            # Load Bayesian HMM model
            bhmm_path = os.path.join(self.model_dir, "bhmm_model.joblib")
            if os.path.exists(bhmm_path):
                self.bhmm_model = joblib.load(bhmm_path)
            
            # Load Particle Filter
            particle_path = os.path.join(self.model_dir, "particle_filter.joblib")
            if os.path.exists(particle_path):
                self.particle_filter = joblib.load(particle_path)
            
            # Load GMM model
            gmm_path = os.path.join(self.model_dir, "gmm_model.joblib")
            if os.path.exists(gmm_path):
                self.gmm_model = joblib.load(gmm_path)
            
            # Load K-Means model
            kmeans_path = os.path.join(self.model_dir, "kmeans_model.joblib")
            if os.path.exists(kmeans_path):
                self.kmeans_model = joblib.load(kmeans_path)
            
            # Load feature names
            features_path = os.path.join(self.model_dir, "features.joblib")
            if os.path.exists(features_path):
                self.selected_features = joblib.load(features_path)
            
            # Load regime characteristics
            char_path = os.path.join(self.model_dir, "regime_characteristics.joblib")
            if os.path.exists(char_path):
                self.regime_characteristics = joblib.load(char_path)
            
            logger.info(f"Models loaded from {self.model_dir}")
            
            return (self.hmm_model is not None or 
                    hasattr(self, 'bhmm_model') and self.bhmm_model is not None or 
                    hasattr(self, 'particle_filter') and self.particle_filter is not None or 
                    self.gmm_model is not None or 
                    self.kmeans_model is not None)
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_current_regime(self, data):
        """
        Identify the current market regime from the provided data.
        
        Args:
            data: DataFrame with market data (recent data)
            
        Returns:
            int: Regime identifier
        """
        try:
            # Extract last data point
            last_data = data.tail(1)
            
            # Predict regime
            regime = self.predict_regimes(last_data)[0]
            
            return int(regime)
            
        except Exception as e:
            logger.error(f"Error getting current regime: {e}")
            return 0  # Default to regime 0
    
    def get_regime_characteristics(self, regime=None):
        """
        Get characteristics of the specified regime or all regimes.
        
        Args:
            regime: Specific regime to get characteristics for (None for all)
            
        Returns:
            dict: Regime characteristics
        """
        if self.regime_characteristics is None:
            return {}
        
        if regime is not None:
            return self.regime_characteristics.get(regime, {})
        else:
            return self.regime_characteristics
    
    def get_regime_description(self, regime):
        """
        Get human-readable description of a regime.
        
        Args:
            regime: Regime identifier
            
        Returns:
            str: Description of the regime
        """
        if self.regime_characteristics is None or regime not in self.regime_characteristics:
            return "Unknown regime"
        
        chars = self.regime_characteristics[regime]
        
        # Determine regime type based on characteristics
        if 'avg_return' in chars and 'volatility' in chars:
            avg_return = chars['avg_return']
            volatility = chars['volatility']
            
            if avg_return > 0.001:  # Positive returns
                if volatility < 0.01:
                    regime_type = "Low-Volatility Bullish"
                elif volatility < 0.02:
                    regime_type = "Moderate-Volatility Bullish"
                else:
                    regime_type = "High-Volatility Bullish"
            elif avg_return < -0.001:  # Negative returns
                if volatility < 0.01:
                    regime_type = "Low-Volatility Bearish"
                elif volatility < 0.02:
                    regime_type = "Moderate-Volatility Bearish"
                else:
                    regime_type = "High-Volatility Bearish"
            else:  # Near-zero returns
                if volatility < 0.005:
                    regime_type = "Stable Sideways"
                elif volatility < 0.015:
                    regime_type = "Choppy Sideways"
                else:
                    regime_type = "Volatile Sideways"
        else:
            regime_type = f"Regime {regime}"
        
        return regime_type
    
    def plot_regime_transitions(self, data, save_path=None):
        """
        Plot transitions between market regimes over time.
        
        Args:
            data: DataFrame with market data
            save_path: Path to save the plot (None to display)
            
        Returns:
            bool: Success or failure
        """
        try:
            # Add timestamp if not present
            if 'Timestamp' not in data.columns and 'Date' in data.columns:
                data['Timestamp'] = data['Date']
            
            if 'Timestamp' not in data.columns:
                logger.warning("No timestamp column found for plotting")
                return False
            
            # Predict regimes
            regimes = self.predict_regimes(data)
            
            # Add to dataframe
            plot_data = pd.DataFrame({
                'Timestamp': data['Timestamp'],
                'Close': data['Close'] if 'Close' in data.columns else np.arange(len(data)),
                'Regime': regimes
            })
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot price series
            ax1.plot(plot_data['Timestamp'], plot_data['Close'], color='blue')
            ax1.set_title('Price Series with Market Regimes')
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            
            # Plot regime transitions
            ax2.scatter(plot_data['Timestamp'], plot_data['Regime'], c=plot_data['Regime'], 
                        cmap='viridis', s=10, alpha=0.7)
            
            # Add regime descriptions
            if self.regime_characteristics:
                unique_regimes = sorted(plot_data['Regime'].unique())
                labels = [self.get_regime_description(regime) for regime in unique_regimes]
                ax2.set_yticks(unique_regimes)
                ax2.set_yticklabels(labels)
            
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Regime')
            ax2.grid(True, alpha=0.3)
            
            # Highlight regime areas
            for regime in sorted(plot_data['Regime'].unique()):
                regime_data = plot_data[plot_data['Regime'] == regime]
                if len(regime_data) > 0:
                    # Get regime color from colormap
                    import matplotlib.cm as cm
                    cmap = cm.get_cmap('viridis')
                    color = cmap(regime / (self.n_regimes - 1)) if self.n_regimes > 1 else cmap(0.5)
                    
                    # Identify contiguous blocks
                    regime_data = regime_data.reset_index()
                    regime_data['block'] = (regime_data['index'] - regime_data.index).diff().ne(0).cumsum()
                    
                    # Highlight each block
                    for block in regime_data['block'].unique():
                        block_data = regime_data[regime_data['block'] == block]
                        if len(block_data) > 1:  # Only shade if at least 2 points
                            start_date = block_data['Timestamp'].iloc[0]
                            end_date = block_data['Timestamp'].iloc[-1]
                            
                            # Highlight in price chart
                            ax1.axvspan(start_date, end_date, color=color, alpha=0.1)
            
            plt.tight_layout()
            
            # Save or display plot
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
            
            return True
            
        except Exception as e:
            logger.error(f"Error plotting regime transitions: {e}")
            return False
    
    def plot_regime_characteristics(self, save_path=None):
        """
        Plot characteristics of each regime.
        
        Args:
            save_path: Path to save the plot (None to display)
            
        Returns:
            bool: Success or failure
        """
        try:
            if not self.regime_characteristics:
                logger.warning("No regime characteristics available for plotting")
                return False
            
            # Create metrics for plotting
            metrics = [
                'avg_return', 
                'volatility', 
                'sharpe',
                'win_rate',
                'avg_duration',
                'pct_in_regime'
            ]
            
            available_metrics = []
            for metric in metrics:
                if all(metric in chars for chars in self.regime_characteristics.values()):
                    available_metrics.append(metric)
            
            if not available_metrics:
                logger.warning("No common metrics found for all regimes")
                return False
            
            # Create data for plotting
            plot_data = pd.DataFrame({
                metric: [self.regime_characteristics[regime].get(metric, 0) 
                        for regime in sorted(self.regime_characteristics.keys())]
                for metric in available_metrics
            })
            
            # Add regime descriptions
            plot_data['Regime'] = [self.get_regime_description(regime) 
                                 for regime in sorted(self.regime_characteristics.keys())]
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(len(available_metrics), 1, figsize=(10, 2*len(available_metrics)))
            
            # Plot each metric
            for i, metric in enumerate(available_metrics):
                ax = axes[i] if len(available_metrics) > 1 else axes
                
                # Use appropriate scale for the metric
                if metric == 'avg_return':
                    plot_data[metric] = plot_data[metric] * 100  # Convert to percentage
                    metric_name = 'Average Daily Return (%)'
                elif metric == 'volatility':
                    plot_data[metric] = plot_data[metric] * 100  # Convert to percentage
                    metric_name = 'Volatility (%)'
                elif metric == 'win_rate':
                    plot_data[metric] = plot_data[metric] * 100  # Convert to percentage
                    metric_name = 'Win Rate (%)'
                elif metric == 'pct_in_regime':
                    plot_data[metric] = plot_data[metric] * 100  # Convert to percentage
                    metric_name = 'Percentage of Time (%)'
                else:
                    metric_name = ' '.join(word.capitalize() for word in metric.split('_'))
                
                # Create bar color based on metric value
                if metric in ['avg_return', 'sharpe', 'win_rate']:
                    # Green for positive, red for negative
                    colors = ['green' if x > 0 else 'red' for x in plot_data[metric]]
                elif metric == 'volatility':
                    # Red for high volatility, green for low
                    colors = ['red' if x > plot_data[metric].median() else 'green' for x in plot_data[metric]]
                else:
                    # Use default color
                    colors = 'blue'
                
                # Plot bars
                bars = ax.bar(plot_data['Regime'], plot_data[metric], color=colors)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(plot_data[metric]),
                            f'{height:.2f}', ha='center', va='bottom', fontsize=9)
                
                ax.set_title(metric_name)
                ax.set_ylabel(metric_name)
                plt.xticks(rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save or display plot
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
            
            return True
            
        except Exception as e:
            logger.error(f"Error plotting regime characteristics: {e}")
            return False


# Demo usage
if __name__ == "__main__":
    import yfinance as yf
    from advanced_feature_engineering import AdvancedFeatureEngineering
    
    # Download sample data
    print("Downloading sample data...")
    data = yf.download("^NDX", period="2y")
    data.reset_index(inplace=True)
    
    # Add features
    print("Adding features...")
    fe = AdvancedFeatureEngineering()
    enhanced_data = fe.add_all_features(data)
    
    # Initialize regime detector with dynamic regime selection
    detector = MarketRegimeDetector(n_regimes=0)  # Set to 0 for automatic regime selection
    
    # Train detector using Bayesian HMM
    print("Training market regime detector using Bayesian HMM...")
    result = detector.fit([enhanced_data], method="bhmm")
    print(f"Training result: {result}")
    
    # Get current regime
    last_data = enhanced_data.tail(5)
    current_regime = detector.get_current_regime(last_data)
    regime_desc = detector.get_regime_description(current_regime)
    
    print(f"\nCurrent market regime: {regime_desc} (ID: {current_regime})")
    
    # Get regime characteristics
    chars = detector.get_regime_characteristics(current_regime)
    print("\nRegime characteristics:")
    for key, value in chars.items():
        print(f"- {key}: {value}")
    
    # Plot regimes
    print("\nPlotting regime transitions...")
    detector.plot_regime_transitions(enhanced_data)
    
    # Plot regime characteristics
    print("\nPlotting regime characteristics...")
    detector.plot_regime_characteristics()
    
    # Try particle filter
    print("\nTraining market regime detector using Particle Filter...")
    detector_pf = MarketRegimeDetector(n_regimes=3)
    result_pf = detector_pf.fit([enhanced_data], method="particle_filter")
    print(f"Training result: {result_pf}")
    
    # Compare results
    print("\nComparing regime transitions between models...")
    bhmm_regimes = detector.predict_regimes(enhanced_data)
    pf_regimes = detector_pf.predict_regimes(enhanced_data)
    
    # Calculate agreement
    agreement = np.mean(bhmm_regimes == pf_regimes)
    print(f"Agreement between BHMM and Particle Filter: {agreement:.2%}")