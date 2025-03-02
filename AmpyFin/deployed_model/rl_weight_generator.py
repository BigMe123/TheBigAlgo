# rl_weight_generator.py
import numpy as np
import pandas as pd
import os
import logging
import time
import json
from datetime import datetime
from collections import deque
import random
import joblib

# Check for tensorflow and gym/gymnasium
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model  # type: ignore
    from tensorflow.keras.layers import Dense, Input, concatenate # type: ignore
    from tensorflow.keras.optimizers import Adam # type: ignore
    
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    
try:
    # Try newer Gymnasium first
    import gymnasium as gym
    from gymnasium import spaces
    GYM_TYPE = "gymnasium"
except ImportError:
    try:
        # Fall back to older gym
        import gym
        from gym import spaces
        GYM_TYPE = "gym"
    except ImportError:
        GYM_TYPE = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketEnvironment:
    """
    Custom market environment for RL training.
    Simulates a trading environment for evaluating weight strategies.
    """
    
    def __init__(self, data):
        """
        Initialize market environment with historical data.
        
        Args:
            data: DataFrame with historical market data
        """
        self.data = data
        self.reset()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # State space: market features + position
        n_features = min(30, len(self.data.columns) - 5)  # Limit to reasonable number
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(n_features + 1,), dtype=np.float32
        )
        
        # Select important features for state
        self.feature_cols = self._select_important_features(n_features)
        
        # Normalization parameters
        self._calculate_normalization_params()
    
    def _select_important_features(self, n_features):
        """Select the most important features for the state representation."""
        # Exclude non-feature columns
        exclude_cols = ['Symbol', 'Timestamp', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Label']
        candidate_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        # If we have too many features, select a subset
        if len(candidate_cols) > n_features:
            # Prefer features related to market regime, technical indicators, and momentum
            priority_features = [col for col in candidate_cols if any(x in col.lower() for x in [
                'regime', 'trend', 'rsi', 'macd', 'momentum', 'volatility', 'sentiment',
                'order', 'flow', 'zscore', 'cross'
            ])]
            
            # Take all priority features
            selected_features = priority_features[:n_features]
            
            # If we need more features, add some from remaining columns
            if len(selected_features) < n_features:
                remaining_features = [col for col in candidate_cols if col not in selected_features]
                additional_features = remaining_features[:n_features - len(selected_features)]
                selected_features.extend(additional_features)
            
            return selected_features
        else:
            return candidate_cols
    
    def _calculate_normalization_params(self):
        """Calculate parameters for normalizing feature values."""
        self.feature_means = {}
        self.feature_stds = {}
        
        for col in self.feature_cols:
            self.feature_means[col] = self.data[col].mean()
            std = self.data[col].std()
            self.feature_stds[col] = std if std > 0 else 1.0
    
    def reset(self):
        """Reset the environment to the beginning of the data."""
        self.current_step = 0
        self.current_position = 0  # No position
        self.positions_history = []
        self.rewards_history = []
        
        # Get initial state
        state = self._get_state()
        
        if GYM_TYPE == "gymnasium":
            return state, {}  # State and info dict for gymnasium
        else:
            return state  # Just state for older gym
    
    def _get_state(self):
        """Get the current state representation."""
        if self.current_step >= len(self.data):
            return np.zeros(len(self.feature_cols) + 1)
        
        # Get features for current step
        features = []
        for col in self.feature_cols:
            # Normalize feature
            value = self.data.iloc[self.current_step][col]
            normalized = (value - self.feature_means[col]) / self.feature_stds[col]
            features.append(normalized)
        
        # Add current position to state
        features.append(self.current_position)
        
        return np.array(features, dtype=np.float32)
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action: Continuous action representing weight adjustment (-1 to 1)
            
        Returns:
            next_state, reward, done, info (or truncated, info for gymnasium)
        """
        # Convert action to position adjustment
        weight_adjustment = float(action[0])  # Action is in [-1, 1]
        
        # Update position based on weight
        # Target position is between -1 (fully short) and 1 (fully long)
        target_position = np.clip(weight_adjustment, -1, 1)
        
        # Calculate actual position change
        # Limit the rate of position change to simulate transaction costs and slippage
        position_change = np.clip(target_position - self.current_position, -0.5, 0.5)
        new_position = self.current_position + position_change
        
        # Record the position
        self.positions_history.append(new_position)
        self.current_position = new_position
        
        # Move to next step
        self.current_step += 1
        
        # Get next state
        next_state = self._get_state()
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Calculate reward
        reward = self._calculate_reward()
        self.rewards_history.append(reward)
        
        # Info dictionary
        info = {
            'position': self.current_position,
            'step': self.current_step
        }
        
        if GYM_TYPE == "gymnasium":
            # For gymnasium, we need to return truncated flag
            return next_state, reward, done, False, info
        else:
            return next_state, reward, done, info
    
    def _calculate_reward(self):
        """
        Calculate the reward for the current action.
        
        Returns:
            float: Reward value
        """
        if self.current_step <= 0 or self.current_step >= len(self.data):
            return 0
        
        # Get price change
        prev_close = self.data.iloc[self.current_step - 1]['Close']
        current_close = self.data.iloc[self.current_step]['Close']
        price_change = (current_close - prev_close) / prev_close
        
        # Position-based reward (profit/loss)
        position_reward = self.current_position * price_change * 100  # Scale up for better learning
        
        # Add risk penalty for large positions
        risk_penalty = -0.1 * abs(self.current_position) ** 2
        
        # Add position change penalty (transaction costs)
        if len(self.positions_history) > 1:
            position_change = abs(self.current_position - self.positions_history[-2])
            transaction_penalty = -0.1 * position_change
        else:
            transaction_penalty = 0
        
        # Combine rewards
        reward = position_reward + risk_penalty + transaction_penalty
        
        return reward
    
    def render(self):
        """Render the environment (placeholder)."""
        pass


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) agent for continuous action space.
    """
    
    def __init__(self, state_dim, action_dim, model_dir="./rl_models"):
        """
        Initialize the DDPG agent.
        
        Args:
            state_dim: Dimensionality of the state space
            action_dim: Dimensionality of the action space
            model_dir: Directory to save/load models
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_dir = model_dir
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.tau = 0.001   # Target network update rate
        self.buffer_size = 100000
        self.exploration_noise = 0.3
        
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
        # Create actor and critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # Create target networks
        self.target_actor = self._build_actor()
        self.target_critic = self._build_critic()
        
        # Initialize target networks with actor/critic weights
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        # Create directories
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def _build_actor(self):
        """
        Build the actor network (policy).
        
        Returns:
            tf.keras.Model: Actor model
        """
        inputs = Input(shape=(self.state_dim,))
        
        x = Dense(256, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        
        # Output layer with tanh activation to bound actions
        outputs = Dense(self.action_dim, activation='tanh')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001))
        
        return model
    
    def _build_critic(self):
        """
        Build the critic network (value function).
        
        Returns:
            tf.keras.Model: Critic model
        """
        # State input
        state_input = Input(shape=(self.state_dim,))
        state_out = Dense(256, activation='relu')(state_input)
        
        # Action input
        action_input = Input(shape=(self.action_dim,))
        
        # Merge state and action
        merged = concatenate([state_out, action_input])
        
        x = Dense(128, activation='relu')(merged)
        x = Dense(64, activation='relu')(x)
        
        # Output Q-value
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=[state_input, action_input], outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model
    
    def get_action(self, state, add_noise=True):
        """
        Get action from the actor network.
        
        Args:
            state: Current state
            add_noise: Whether to add exploration noise
            
        Returns:
            numpy.ndarray: Action
        """
        state = np.reshape(state, [1, self.state_dim])
        action = self.actor.predict(state)[0]
        
        if add_noise:
            noise = self.exploration_noise * np.random.randn(self.action_dim)
            action = np.clip(action + noise, -1, 1)
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        """
        Train the agent by sampling from replay buffer.
        
        Returns:
            float: Critic loss
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        # Sample random batch from replay buffer
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i in indices:
            states.append(self.replay_buffer[i][0])
            actions.append(self.replay_buffer[i][1])
            rewards.append(self.replay_buffer[i][2])
            next_states.append(self.replay_buffer[i][3])
            dones.append(self.replay_buffer[i][4])
        
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Get target actions from target actor
        target_actions = self.target_actor.predict(next_states)
        
        # Get target Q-values from target critic
        target_q = self.target_critic.predict([next_states, target_actions])
        
        # Compute target values
        target = rewards + self.gamma * target_q * (1 - dones)
        
        # Train critic
        critic_loss = self.critic.train_on_batch([states, actions], target)
        
        # Train actor
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, actions]))
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Update target networks
        self._update_target_networks()
        
        return critic_loss
    
    def _update_target_networks(self):
        """Update target networks with Polyak averaging."""
        # Update target actor
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        
        for i in range(len(actor_weights)):
            target_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * target_actor_weights[i]
        
        self.target_actor.set_weights(target_actor_weights)
        
        # Update target critic
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        
        for i in range(len(critic_weights)):
            target_critic_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * target_critic_weights[i]
        
        self.target_critic.set_weights(target_critic_weights)
    
    def save_models(self):
        """Save actor and critic models."""
        self.actor.save(os.path.join(self.model_dir, 'actor.h5'))
        self.critic.save(os.path.join(self.model_dir, 'critic.h5'))
        logger.info(f"Models saved to {self.model_dir}")
    
    def load_models(self):
        """Load actor and critic models if they exist."""
        actor_path = os.path.join(self.model_dir, 'actor.h5')
        critic_path = os.path.join(self.model_dir, 'critic.h5')
        
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            try:
                self.actor = load_model(actor_path)
                self.critic = load_model(critic_path)
                self.target_actor = load_model(actor_path)
                self.target_critic = load_model(critic_path)
                logger.info("Models loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading models: {e}")
                return False
        else:
            logger.info("No saved models found")
            return False


class RLWeightGenerator:
    """
    Reinforcement Learning-based weight generator for trading decisions.
    Uses DDPG for continuous action space to optimize weight generation.
    """
    
    def __init__(self, model_dir="./rl_weights"):
        """
        Initialize the RL weight generator.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = model_dir
        self.agent = None
        self.env = None
        self.feature_cols = None
        self.feature_means = None
        self.feature_stds = None
        
        # Training parameters
        self.episodes = 100
        self.max_steps = 1000
        
        # Market regime adaptation
        self.regime_models = {}
        self.current_regime = None
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Check if RL libraries are available
        if not RL_AVAILABLE or GYM_TYPE is None:
            logger.warning("TensorFlow or Gym/Gymnasium not available, RL weight generation disabled")
    
    def check_rl_availability(self):
        """Check if RL libraries are available."""
        return RL_AVAILABLE and GYM_TYPE is not None
    
    def is_trained(self):
        """Check if models are trained."""
        agent_file = os.path.join(self.model_dir, 'agent_metadata.json')
        actor_file = os.path.join(self.model_dir, 'rl_models', 'actor.h5')
        
        return os.path.exists(agent_file) and os.path.exists(actor_file)
    
    def train(self, data_frames, epochs=5, save_freq=10):
        """
        Train the RL agent using historical data.
        
        Args:
            data_frames: List of pandas DataFrames with historical data
            epochs: Number of training epochs (each epoch runs through all data)
            save_freq: Frequency to save models (in episodes)
            
        Returns:
            dict: Training metrics
        """
        if not self.check_rl_availability():
            return {"success": False, "message": "RL libraries not available"}
        
        try:
            # Set TF memory growth to avoid OOM errors
            try:
                physical_devices = tf.config.list_physical_devices('GPU')
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
            except:
                pass
            
            # Concatenate all data frames
            all_data = pd.concat(data_frames, ignore_index=True)
            
            # Ensure we have required columns
            required_cols = ['Close', 'Volume', 'Label']
            if not all(col in all_data.columns for col in required_cols):
                return {"success": False, "message": "Missing required columns in data"}
            
            # Look for market regime column
            regime_col = None
            for col in all_data.columns:
                if 'regime' in col.lower() or 'state' in col.lower():
                    regime_col = col
                    break
            
            # If market regime is available, train separate models for each regime
            if regime_col is not None:
                # Get unique regimes
                regimes = all_data[regime_col].unique()
                
                logger.info(f"Training separate models for {len(regimes)} market regimes")
                
                regime_metrics = {}
                
                for regime in regimes:
                    # Get data for this regime
                    regime_data = all_data[all_data[regime_col] == regime].copy()
                    
                    if len(regime_data) < 100:
                        logger.warning(f"Not enough data for regime {regime}, skipping")
                        continue
                    
                    logger.info(f"Training model for regime {regime} with {len(regime_data)} samples")
                    
                    # Train model for this regime
                    metrics = self._train_single_model(
                        regime_data, 
                        model_dir=os.path.join(self.model_dir, f"regime_{regime}"),
                        epochs=epochs,
                        save_freq=save_freq
                    )
                    
                    regime_metrics[f"regime_{regime}"] = metrics
                
                # Save metadata
                self._save_metadata({"regimes": True, "regime_col": regime_col, "regimes": regimes.tolist()})
                
                return {"success": True, "message": "Regime-based models trained", "metrics": regime_metrics}
            else:
                # Train single model for all data
                metrics = self._train_single_model(all_data, epochs=epochs, save_freq=save_freq)
                
                # Save metadata
                self._save_metadata({"regimes": False})
                
                return {"success": True, "message": "Model trained successfully", "metrics": metrics}
                
        except Exception as e:
            logger.error(f"Error training RL model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def _train_single_model(self, data, model_dir=None, epochs=5, save_freq=10):
        """
        Train a single RL model on the provided data.
        
        Args:
            data: DataFrame with historical data
            model_dir: Directory to save model
            epochs: Number of training epochs
            save_freq: Frequency to save models
            
        Returns:
            dict: Training metrics
        """
        # Use class model_dir if not specified
        model_dir = model_dir or os.path.join(self.model_dir, "rl_models")
        
        # Sort data by timestamp if available
        if 'Timestamp' in data.columns:
            data = data.sort_values('Timestamp').reset_index(drop=True)
        
        # Initialize environment
        self.env = MarketEnvironment(data)
        
        # Extract feature information
        self.feature_cols = self.env.feature_cols
        self.feature_means = self.env.feature_means
        self.feature_stds = self.env.feature_stds
        
        # Save feature information
        feature_info = {
            "feature_cols": self.feature_cols,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds
        }
        
        with open(os.path.join(model_dir, "feature_info.json"), 'w') as f:
            # Convert dict values to serializable types
            serializable_info = {
                "feature_cols": self.feature_cols,
                "feature_means": {k: float(v) for k, v in self.feature_means.items()},
                "feature_stds": {k: float(v) for k, v in self.feature_stds.items()}
            }
            json.dump(serializable_info, f)
        
        # Initialize agent
        state_dim = len(self.env.feature_cols) + 1  # Features + position
        action_dim = 1  # Continuous weight adjustment
        
        self.agent = DDPGAgent(state_dim, action_dim, model_dir=model_dir)
        
        # Training metrics
        metrics = {
            "rewards": [],
            "steps": [],
            "losses": []
        }
        
        # Training loop
        total_episodes = epochs * (len(data) // self.env.observation_space.shape[0])
        total_episodes = max(total_episodes, 50)  # Ensure at least 50 episodes
        
        logger.info(f"Training RL agent for {total_episodes} episodes...")
        
        for episode in range(total_episodes):
            # Reset environment
            if GYM_TYPE == "gymnasium":
                state, _ = self.env.reset()
            else:
                state = self.env.reset()
            
            episode_reward = 0
            episode_steps = 0
            episode_losses = []
            
            # Run episode
            done = False
            while not done:
                # Get action
                action = self.agent.get_action(state)
                
                # Take step
                if GYM_TYPE == "gymnasium":
                    next_state, reward, done, _, _ = self.env.step(action)
                else:
                    next_state, reward, done, _ = self.env.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                loss = self.agent.train()
                episode_losses.append(loss)
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # Limit episode length
                if episode_steps >= self.max_steps:
                    break
            
            # Update metrics
            metrics["rewards"].append(episode_reward)
            metrics["steps"].append(episode_steps)
            metrics["losses"].append(np.mean(episode_losses) if episode_losses else 0)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                logger.info(f"Episode {episode + 1}/{total_episodes}, Reward: {episode_reward:.2f}, Steps: {episode_steps}")
            
            # Save models periodically
            if (episode + 1) % save_freq == 0:
                self.agent.save_models()
        
        # Save final models
        self.agent.save_models()
        
        logger.info("RL agent training completed")
        
        return metrics
    
    def _save_metadata(self, metadata):
        """Save metadata about the trained models."""
        metadata["last_updated"] = datetime.now().isoformat()
        metadata["version"] = "1.0"
        
        with open(os.path.join(self.model_dir, "agent_metadata.json"), 'w') as f:
            json.dump(metadata, f)
    
    def load_models(self):
        """Load trained models."""
        if not self.check_rl_availability():
            return False
        
        try:
            # Check if metadata file exists
            metadata_file = os.path.join(self.model_dir, "agent_metadata.json")
            
            if not os.path.exists(metadata_file):
                logger.warning("No metadata file found, cannot load models")
                return False
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if we have regime-based models
            if metadata.get("regimes", False):
                logger.info("Loading regime-based models")
                
                # Get regime column
                self.regime_col = metadata.get("regime_col")
                
                # Load each regime model
                for regime in metadata.get("regimes", []):
                    regime_dir = os.path.join(self.model_dir, f"regime_{regime}")
                    
                    # Load feature info
                    feature_file = os.path.join(regime_dir, "feature_info.json")
                    
                    if os.path.exists(feature_file):
                        with open(feature_file, 'r') as f:
                            feature_info = json.load(f)
                            
                        # Initialize agent
                        state_dim = len(feature_info["feature_cols"]) + 1  # Features + position
                        action_dim = 1
                        
                        agent = DDPGAgent(state_dim, action_dim, model_dir=regime_dir)
                        if agent.load_models():
                            # Store agent and feature info
                            self.regime_models[str(regime)] = {
                                "agent": agent,
                                "feature_cols": feature_info["feature_cols"],
                                "feature_means": feature_info["feature_means"],
                                "feature_stds": feature_info["feature_stds"]
                            }
                            logger.info(f"Loaded model for regime {regime}")
                
                return len(self.regime_models) > 0
                
            else:
                logger.info("Loading single model")
                
                # Load feature info
                feature_file = os.path.join(self.model_dir, "rl_models", "feature_info.json")
                
                if not os.path.exists(feature_file):
                    logger.warning("Feature info file not found")
                    return False
                
                with open(feature_file, 'r') as f:
                    feature_info = json.load(f)
                
                self.feature_cols = feature_info["feature_cols"]
                self.feature_means = feature_info["feature_means"]
                self.feature_stds = feature_info["feature_stds"]
                
                # Initialize agent
                state_dim = len(self.feature_cols) + 1  # Features + position
                action_dim = 1
                
                self.agent = DDPGAgent(state_dim, action_dim, model_dir=os.path.join(self.model_dir, "rl_models"))
                
                return self.agent.load_models()
                
        except Exception as e:
            logger.error(f"Error loading RL models: {e}")
            return False
    
    def generate_weight(self, data, base_weight=500, position=0):
        """
        Generate trading weight using the trained RL model.
        
        Args:
            data: DataFrame with market data
            base_weight: Base weight to adjust
            position: Current position (for state representation)
            
        Returns:
            tuple: (buy_weight, sell_weight)
        """
        if not self.check_rl_availability():
            logger.warning("RL libraries not available, using default weights")
            return base_weight, 0
        
        try:
            # Check if models are loaded
            if not self.agent and not self.regime_models:
                success = self.load_models()
                if not success:
                    logger.warning("Could not load RL models, using default weights")
                    return base_weight, 0
            
            # Determine market regime if using regime-based models
            regime = None
            if self.regime_models and hasattr(self, 'regime_col') and self.regime_col in data.columns:
                regime = str(data[self.regime_col].iloc[-1])
                
                # Use appropriate model for this regime
                if regime in self.regime_models:
                    logger.info(f"Using model for regime {regime}")
                    model_info = self.regime_models[regime]
                    agent = model_info["agent"]
                    feature_cols = model_info["feature_cols"]
                    feature_means = model_info["feature_means"]
                    feature_stds = model_info["feature_stds"]
                else:
                    # Fallback to first available model
                    logger.warning(f"No model for regime {regime}, using default")
                    regime_key = list(self.regime_models.keys())[0]
                    model_info = self.regime_models[regime_key]
                    agent = model_info["agent"]
                    feature_cols = model_info["feature_cols"]
                    feature_means = model_info["feature_means"]
                    feature_stds = model_info["feature_stds"]
            else:
                # Use single model
                agent = self.agent
                feature_cols = self.feature_cols
                feature_means = self.feature_means
                feature_stds = self.feature_stds
            
            # Ensure we have all needed columns
            missing_cols = [col for col in feature_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"Missing columns in data: {missing_cols}")
                return base_weight, 0
            
            # Prepare state
            state = []
            for col in feature_cols:
                value = data[col].iloc[-1]
                normalized = (value - feature_means[col]) / feature_stds[col]
                state.append(normalized)
            
            # Add position
            state.append(position)
            
            # Get action from agent
            action = agent.get_action(np.array(state), add_noise=False)
            
            # Convert action to weight
            # Action is in [-1, 1], need to convert to buy/sell weights
            action_value = float(action[0])
            
            # Scale action to weights
            if action_value > 0:  # Buy signal
                buy_weight = base_weight * action_value
                sell_weight = 0
            else:  # Sell signal
                buy_weight = 0
                sell_weight = base_weight * -action_value
            
            logger.info(f"RL weight: action={action_value:.2f}, buy={buy_weight:.2f}, sell={sell_weight:.2f}")
            
            return buy_weight, sell_weight
            
        except Exception as e:
            logger.error(f"Error generating weight: {e}")
            return base_weight, 0


# Demo usage
if __name__ == "__main__":
    import yfinance as yf
    from advanced_feature_engineering import AdvancedFeatureEngineering
    
    # Check if RL is available
    if not RL_AVAILABLE or GYM_TYPE is None:
        print("TensorFlow or Gym/Gymnasium not available. Install them to use RL weight generation.")
        import sys
        sys.exit(0)
    
    # Download sample data
    print("Downloading sample data...")
    data = yf.download("AAPL", period="1y")
    data["Symbol"] = "AAPL"
    data.reset_index(inplace=True)
    
    # Add label (1 if price increases)
    data['Label'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Add advanced features
    print("Adding features...")
    fe = AdvancedFeatureEngineering()
    enhanced_data = fe.add_all_features(data)
    
    # Initialize RL weight generator
    rl_generator = RLWeightGenerator(model_dir="./rl_weights")
    
    # Train or load model
    if not rl_generator.is_trained():
        print("Training RL model...")
        result = rl_generator.train([enhanced_data], epochs=2, save_freq=5)
        print(f"Training result: {result}")
    else:
        print("Loading existing RL model...")
        rl_generator.load_models()
    
    # Generate weight for recent data
    last_data = enhanced_data.tail(1)
    buy_weight, sell_weight = rl_generator.generate_weight(last_data)
    
    print(f"\nGenerated weights:")
    print(f"Buy weight: {buy_weight:.2f}")
    print(f"Sell weight: {sell_weight:.2f}")
    
    if buy_weight > sell_weight:
        print("Recommendation: BUY")
    elif sell_weight > buy_weight:
        print("Recommendation: SELL")
    else:
        print("Recommendation: HOLD")