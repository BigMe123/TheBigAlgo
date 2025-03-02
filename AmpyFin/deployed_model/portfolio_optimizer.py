# portfolio_optimizer.py
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm, t

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Smart portfolio optimization and risk management module.
    
    Provides advanced portfolio construction, asset allocation, and risk management
    based on modern portfolio theory, Kelly criterion, and drawdown constraints.
    """
    
    def __init__(self, risk_free_rate=0.03, max_drawdown_limit=0.15):
        """
        Initialize the portfolio optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate (decimal)
            max_drawdown_limit: Maximum acceptable drawdown (decimal)
        """
        self.risk_free_rate = risk_free_rate
        self.max_drawdown_limit = max_drawdown_limit
        
        # Portfolio constraints
        self.max_position_size = 0.20  # Maximum 20% in any single position
        self.min_position_size = 0.01  # Minimum 1% for any included position
        self.min_stocks = 5           # Minimum number of stocks to include
        
        # Sector constraints
        self.max_sector_allocation = 0.40  # Maximum 40% in any sector
        
        # Asset universe
        self.asset_data = None
        self.returns = None
        self.cov_matrix = None
        self.expected_returns = None
        self.sector_map = None
        
        # Optimization results
        self.optimal_weights = None
        self.efficient_frontier = None
        self.portfolio_metrics = None
    
    def load_market_data(self, price_data, sector_data=None):
        """
        Load market data for portfolio optimization.
        
        Args:
            price_data: DataFrame with price data (time x assets)
            sector_data: DataFrame or dict mapping assets to sectors
            
        Returns:
            bool: Success or failure
        """
        try:
            # Store asset data
            self.asset_data = price_data.copy()
            
            # Calculate returns
            self.returns = price_data.pct_change().dropna()
            
            # Calculate covariance matrix and expected returns
            # Use shrinkage to improve stability
            self.cov_matrix = self._calculate_shrinkage_covariance(self.returns)
            
            # Use multiple methods for expected returns
            self.expected_returns = self._calculate_expected_returns(self.returns)
            
            # Store sector data if provided
            if sector_data is not None:
                self.sector_map = sector_data if isinstance(sector_data, dict) else sector_data.to_dict()
            else:
                self.sector_map = {col: 'Unknown' for col in price_data.columns}
            
            logger.info(f"Loaded market data with {len(self.returns.columns)} assets and {len(self.returns)} time periods")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _calculate_shrinkage_covariance(self, returns):
        """
        Calculate covariance matrix with shrinkage for stability.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            DataFrame: Shrinkage-adjusted covariance matrix
        """
        # Calculate sample covariance matrix
        sample_cov = returns.cov()
        
        # Calculate target (diagonal matrix with sample variances)
        target = np.diag(np.diag(sample_cov))
        
        # Compute shrinkage intensity (simplified Ledoit-Wolf)
        # This is a simplified implementation - a full implementation would
        # estimate the optimal shrinkage factor based on data
        shrinkage_factor = 0.2  # Fixed shrinkage for simplicity
        
        # Compute shrinkage estimator
        shrunk_cov = (1 - shrinkage_factor) * sample_cov + shrinkage_factor * target
        
        return pd.DataFrame(shrunk_cov, index=sample_cov.index, columns=sample_cov.columns)
    
    def _calculate_expected_returns(self, returns):
        """
        Calculate expected returns using multiple methods.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Series: Expected returns for each asset
        """
        # Method 1: Historical mean returns
        mean_returns = returns.mean() * 252  # Annualize daily returns
        
        # Method 2: CAPM-adjusted returns
        market_returns = returns.mean(axis=1)  # Use equal-weighted portfolio as market proxy
        
        # Calculate beta for each asset
        betas = {}
        for col in returns.columns:
            cov_with_market = returns[col].cov(market_returns)
            market_var = market_returns.var()
            beta = cov_with_market / market_var if market_var > 0 else 1.0
            betas[col] = beta
        
        # Calculate CAPM expected returns
        market_premium = market_returns.mean() * 252 - self.risk_free_rate
        capm_returns = pd.Series({
            col: self.risk_free_rate + betas[col] * market_premium
            for col in returns.columns
        })
        
        # Method 3: Exponentially weighted returns (recent data has more weight)
        ewm_returns = returns.ewm(halflife=30).mean().iloc[-1] * 252
        
        # Combine methods (weighted average)
        expected_returns = (0.3 * mean_returns + 
                           0.3 * capm_returns + 
                           0.4 * ewm_returns)
        
        return expected_returns
    
    def optimize_portfolio(self, objective='sharpe', risk_aversion=2.0, constraints=None):
        """
        Optimize the portfolio based on the specified objective.
        
        Args:
            objective: Optimization objective (sharpe, return, risk, utility)
            risk_aversion: Risk aversion parameter for utility optimization
            constraints: Additional constraints for the optimization
            
        Returns:
            dict: Optimization results
        """
        try:
            logger.info(f"Optimizing portfolio with objective: {objective}")
            
            # Check if data is loaded
            if self.returns is None or self.cov_matrix is None or self.expected_returns is None:
                logger.error("Market data not loaded. Call load_market_data first.")
                return {"success": False, "message": "Market data not loaded"}
            
            # Handle case with only one asset
            if len(self.returns.columns) == 1:
                asset = self.returns.columns[0]
                self.optimal_weights = pd.Series({asset: 1.0})
                return {
                    "success": True, 
                    "weights": self.optimal_weights,
                    "message": "Only one asset available, assigned 100% weight"
                }
            
            # Define optimization objective
            if objective == 'sharpe':
                opt_function = self._maximize_sharpe
            elif objective == 'return':
                opt_function = self._maximize_return
            elif objective == 'risk':
                opt_function = self._minimize_risk
            elif objective == 'utility':
                # Capture risk_aversion in a closure
                def utility_objective(weights):
                    return self._maximize_utility(weights, risk_aversion)
                opt_function = utility_objective
            else:
                logger.warning(f"Unknown objective: {objective}, defaulting to Sharpe ratio")
                opt_function = self._maximize_sharpe
            
            # Define constraints
            all_constraints = []
            
            # Basic constraint: weights sum to 1
            sum_constraint = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
            all_constraints.append(sum_constraint)
            
            # Sector constraints if sector data is available
            if self.sector_map is not None:
                sectors = set(self.sector_map.values())
                for sector in sectors:
                    # Get indices of assets in this sector
                    sector_indices = [i for i, (asset, s) in enumerate(self.sector_map.items())
                                    if s == sector and asset in self.returns.columns]
                    
                    if sector_indices:
                        # Create a constraint function for this sector
                        def sector_constraint(weights, indices=sector_indices):
                            sector_allocation = sum(weights[i] for i in indices)
                            return self.max_sector_allocation - sector_allocation
                        
                        all_constraints.append({'type': 'ineq', 'fun': sector_constraint})
            
            # Add user-provided constraints
            if constraints:
                all_constraints.extend(constraints)
            
            # Initial weights (equal allocation)
            n_assets = len(self.returns.columns)
            initial_weights = np.ones(n_assets) / n_assets
            
            # Define bounds (min and max position sizes)
            bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
            
            # Run the optimization
            opt_result = minimize(
                opt_function,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=all_constraints,
                options={'disp': False, 'maxiter': 1000}
            )
            
            if opt_result['success']:
                # Get optimized weights
                weights = opt_result['x']
                
                # Ensure no negative weights and normalize
                weights = np.clip(weights, 0, None)
                weights = weights / np.sum(weights)
                
                # Convert to Series
                self.optimal_weights = pd.Series(
                    weights, 
                    index=self.returns.columns
                )
                
                # Calculate portfolio metrics
                self.portfolio_metrics = self._calculate_portfolio_metrics(self.optimal_weights)
                
                logger.info(f"Portfolio optimization successful, Sharpe ratio: {self.portfolio_metrics['sharpe_ratio']:.4f}")
                
                return {
                    "success": True,
                    "weights": self.optimal_weights,
                    "metrics": self.portfolio_metrics
                }
            else:
                logger.error(f"Portfolio optimization failed: {opt_result['message']}")
                return {"success": False, "message": opt_result['message']}
                
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def _maximize_sharpe(self, weights):
        """Negative Sharpe ratio for minimization."""
        portfolio_return, portfolio_risk = self._calculate_portfolio_return_risk(weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        return -sharpe_ratio  # Negative for minimization
    
    def _maximize_return(self, weights):
        """Negative portfolio return for minimization."""
        portfolio_return, _ = self._calculate_portfolio_return_risk(weights)
        return -portfolio_return  # Negative for minimization
    
    def _minimize_risk(self, weights):
        """Portfolio risk for minimization."""
        _, portfolio_risk = self._calculate_portfolio_return_risk(weights)
        return portfolio_risk
    
    def _maximize_utility(self, weights, risk_aversion):
        """
        Negative utility for minimization.
        Utility = Return - (risk_aversion * Variance / 2)
        """
        portfolio_return, portfolio_risk = self._calculate_portfolio_return_risk(weights)
        utility = portfolio_return - (risk_aversion * (portfolio_risk ** 2) / 2)
        return -utility  # Negative for minimization
    
    def _calculate_portfolio_return_risk(self, weights):
        """
        Calculate portfolio expected return and risk.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            tuple: (expected return, risk)
        """
        # Convert weights to numpy array if needed
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        
        # Calculate expected portfolio return
        portfolio_return = np.sum(self.expected_returns.values * weights)
        
        # Calculate portfolio risk (standard deviation)
        portfolio_variance = weights.T @ self.cov_matrix.values @ weights
        portfolio_risk = np.sqrt(portfolio_variance)
        
        return portfolio_return, portfolio_risk
    
    def _calculate_portfolio_metrics(self, weights):
        """
        Calculate comprehensive portfolio metrics.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            dict: Portfolio metrics
        """
        # Convert to numpy for calculations
        if isinstance(weights, pd.Series):
            weights_array = weights.values
        else:
            weights_array = np.array(weights)
        
        # Expected return and risk
        portfolio_return, portfolio_risk = self._calculate_portfolio_return_risk(weights_array)
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        # Portfolio composition
        composition = {}
        sorted_weights = sorted(zip(weights.index, weights), key=lambda x: x[1], reverse=True)
        for asset, weight in sorted_weights:
            if weight >= 0.005:  # Only include positions >= 0.5%
                composition[asset] = weight
        
        # Calculate risk contribution
        risk_contribution = {}
        marginal_contribution = self.cov_matrix @ weights_array
        total_risk_contribution = weights_array * marginal_contribution
        # Normalize by total risk
        if portfolio_risk > 0:
            total_risk_contribution = total_risk_contribution / portfolio_risk
        
        for i, asset in enumerate(self.returns.columns):
            risk_contribution[asset] = total_risk_contribution[i]
        
        # Calculate sector allocation
        sector_allocation = {}
        if self.sector_map:
            for asset, weight in zip(self.returns.columns, weights_array):
                sector = self.sector_map.get(asset, 'Unknown')
                if sector in sector_allocation:
                    sector_allocation[sector] += weight
                else:
                    sector_allocation[sector] = weight
        
        # Calculate historical drawdown with these weights
        historical_returns = self.returns @ weights_array
        cumulative_returns = (1 + historical_returns).cumprod()
        historical_drawdown = (cumulative_returns / cumulative_returns.cummax()) - 1
        max_drawdown = historical_drawdown.min()
        
        # Value at Risk (VaR) and Conditional Value at Risk (CVaR)
        var_95 = norm.ppf(0.05, portfolio_return / 252, portfolio_risk / np.sqrt(252))
        cvar_95 = -portfolio_return + portfolio_risk * norm.pdf(norm.ppf(0.05)) / 0.05
        
        # Kelly criterion for position sizing
        win_probability = (historical_returns > 0).mean()
        avg_win = historical_returns[historical_returns > 0].mean()
        avg_loss = abs(historical_returns[historical_returns < 0].mean())
        kelly_fraction = (win_probability / avg_loss) - ((1 - win_probability) / avg_win) if avg_win > 0 and avg_loss > 0 else 0
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = -portfolio_return / max_drawdown if max_drawdown < 0 else np.inf
        
        metrics = {
            'expected_return': portfolio_return,
            'annual_volatility': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'composition': composition,
            'risk_contribution': risk_contribution,
            'sector_allocation': sector_allocation,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_probability': win_probability,
            'kelly_fraction': kelly_fraction,
            'calmar_ratio': calmar_ratio
        }
        
        return metrics
    
    def generate_efficient_frontier(self, points=50):
        """
        Generate the efficient frontier for the current asset universe.
        
        Args:
            points: Number of points on the frontier
            
        Returns:
            DataFrame: Efficient frontier data
        """
        try:
            logger.info(f"Generating efficient frontier with {points} points")
            
            # Check if data is loaded
            if self.returns is None or self.cov_matrix is None or self.expected_returns is None:
                logger.error("Market data not loaded. Call load_market_data first.")
                return None
            
            # Find the minimum risk portfolio
            n_assets = len(self.returns.columns)
            min_risk_result = minimize(
                self._minimize_risk,
                np.ones(n_assets) / n_assets,
                method='SLSQP',
                bounds=[(self.min_position_size, self.max_position_size) for _ in range(n_assets)],
                constraints={'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                options={'disp': False}
            )
            
            min_risk_weights = min_risk_result['x']
            min_return, min_risk = self._calculate_portfolio_return_risk(min_risk_weights)
            
            # Find the maximum return portfolio
            max_return_result = minimize(
                self._maximize_return,
                np.ones(n_assets) / n_assets,
                method='SLSQP',
                bounds=[(self.min_position_size, self.max_position_size) for _ in range(n_assets)],
                constraints={'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                options={'disp': False}
            )
            
            max_return_weights = max_return_result['x']
            max_return, max_risk = self._calculate_portfolio_return_risk(max_return_weights)
            
            # Generate a range of target returns
            target_returns = np.linspace(min_return, max_return, points)
            risks = []
            sharpes = []
            weights_list = []
            
            # For each target return, find the minimum risk portfolio
            for target_return in target_returns:
                # Add a return constraint
                return_constraint = {
                    'type': 'eq',
                    'fun': lambda weights: self._calculate_portfolio_return_risk(weights)[0] - target_return
                }
                
                # Run optimization
                result = minimize(
                    self._minimize_risk,
                    np.ones(n_assets) / n_assets,
                    method='SLSQP',
                    bounds=[(self.min_position_size, self.max_position_size) for _ in range(n_assets)],
                    constraints=[
                        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                        return_constraint
                    ],
                    options={'disp': False}
                )
                
                if result['success']:
                    weights = result['x']
                    weights = np.clip(weights, 0, None)
                    weights = weights / np.sum(weights)
                    weights_list.append(weights)
                    
                    _, risk = self._calculate_portfolio_return_risk(weights)
                    risks.append(risk)
                    
                    sharpe = (target_return - self.risk_free_rate) / risk if risk > 0 else 0
                    sharpes.append(sharpe)
                else:
                    # If optimization fails, use interpolation
                    weights_list.append(None)
                    risks.append(None)
                    sharpes.append(None)
            
            # Create DataFrame with frontier data
            self.efficient_frontier = pd.DataFrame({
                'return': target_returns,
                'risk': risks,
                'sharpe': sharpes
            })
            
            # Find the maximum Sharpe ratio portfolio on the frontier
            valid_indices = [i for i, s in enumerate(sharpes) if s is not None]
            if valid_indices:
                max_sharpe_idx = valid_indices[np.argmax([sharpes[i] for i in valid_indices])]
                max_sharpe_weights = weights_list[max_sharpe_idx]
                self.efficient_frontier.loc[max_sharpe_idx, 'max_sharpe'] = True
            
            logger.info("Generated efficient frontier successfully")
            
            return self.efficient_frontier
            
        except Exception as e:
            logger.error(f"Error generating efficient frontier: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def optimize_with_kelly(self, leverage_constraint=1.0):
        """
        Optimize the portfolio using Kelly criterion for position sizing.
        
        Args:
            leverage_constraint: Maximum allowed leverage
            
        Returns:
            dict: Optimization results
        """
        try:
            logger.info("Optimizing portfolio with Kelly criterion")
            
            # Check if data is loaded
            if self.returns is None:
                logger.error("Market data not loaded. Call load_market_data first.")
                return {"success": False, "message": "Market data not loaded"}
            
            # Calculate historical mean and covariance
            mu = self.returns.mean() * 252  # Annual returns
            sigma = self.returns.cov() * 252  # Annual covariance
            
            # Calculate full Kelly weights (no constraints)
            # Kelly formula: K = Σ^(-1) * μ 
            try:
                # Use pseudo-inverse for numerical stability
                sigma_inv = pd.DataFrame(
                    np.linalg.pinv(sigma.values),
                    index=sigma.index,
                    columns=sigma.columns
                )
                kelly_weights = sigma_inv @ mu
                
                # Normalize to leverage constraint
                kelly_sum = kelly_weights.sum()
                if kelly_sum > 0:
                    normalized_kelly = kelly_weights * (leverage_constraint / kelly_sum)
                else:
                    normalized_kelly = pd.Series(0, index=kelly_weights.index)
                
                # Apply position constraints
                constrained_kelly = pd.Series({
                    asset: min(max(weight, self.min_position_size), self.max_position_size)
                    for asset, weight in normalized_kelly.items()
                })
                
                # Renormalize to leverage constraint
                constrained_kelly = constrained_kelly * (leverage_constraint / constrained_kelly.sum())
                
                # Calculate half-Kelly weights (more conservative)
                half_kelly = constrained_kelly * 0.5
                
                # Calculate portfolio metrics for both
                full_kelly_metrics = self._calculate_portfolio_metrics(constrained_kelly)
                half_kelly_metrics = self._calculate_portfolio_metrics(half_kelly)
                
                # Store optimal weights (use half-Kelly as default)
                self.optimal_weights = half_kelly
                self.portfolio_metrics = half_kelly_metrics
                
                logger.info(f"Kelly optimization successful, Sharpe ratio: {half_kelly_metrics['sharpe_ratio']:.4f}")
                
                return {
                    "success": True,
                    "full_kelly": constrained_kelly,
                    "half_kelly": half_kelly,
                    "full_kelly_metrics": full_kelly_metrics,
                    "half_kelly_metrics": half_kelly_metrics
                }
                
            except np.linalg.LinAlgError as e:
                logger.error(f"Linear algebra error in Kelly optimization: {e}")
                return {"success": False, "message": f"Linear algebra error: {str(e)}"}
                
        except Exception as e:
            logger.error(f"Error in Kelly optimization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def optimize_with_risk_parity(self):
        """
        Optimize the portfolio using risk parity (equal risk contribution).
        
        Returns:
            dict: Optimization results
        """
        try:
            logger.info("Optimizing portfolio with risk parity")
            
            # Check if data is loaded
            if self.cov_matrix is None:
                logger.error("Market data not loaded. Call load_market_data first.")
                return {"success": False, "message": "Market data not loaded"}
            
            # Define risk contribution objective function
            def risk_budget_objective(weights, cov_matrix, target_risk):
                weights = np.array(weights)
                portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
                risk_contribution = weights * (cov_matrix @ weights) / portfolio_risk
                risk_target = portfolio_risk * target_risk
                return ((risk_contribution - risk_target)**2).sum()
            
            # Number of assets
            n_assets = len(self.cov_matrix.columns)
            
            # Target risk (equal for all assets)
            target_risk = np.ones(n_assets) / n_assets
            
            # Initial weights (equal allocation)
            initial_weights = np.ones(n_assets) / n_assets
            
            # Define bounds
            bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
            
            # Define weight sum constraint
            weight_constraint = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
            
            # Convert covariance matrix to numpy array
            cov_matrix_array = self.cov_matrix.values
            
            # Run optimization
            opt_result = minimize(
                lambda weights: risk_budget_objective(weights, cov_matrix_array, target_risk),
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=[weight_constraint],
                options={'disp': False, 'maxiter': 1000}
            )
            
            if opt_result['success']:
                # Get optimized weights
                weights = opt_result['x']
                
                # Ensure no negative weights and normalize
                weights = np.clip(weights, 0, None)
                weights = weights / np.sum(weights)
                
                # Convert to Series
                self.optimal_weights = pd.Series(weights, index=self.cov_matrix.columns)
                
                # Calculate portfolio metrics
                self.portfolio_metrics = self._calculate_portfolio_metrics(self.optimal_weights)
                
                logger.info(f"Risk parity optimization successful, Sharpe ratio: {self.portfolio_metrics['sharpe_ratio']:.4f}")
                
                return {
                    "success": True,
                    "weights": self.optimal_weights,
                    "metrics": self.portfolio_metrics
                }
            else:
                logger.error(f"Risk parity optimization failed: {opt_result['message']}")
                return {"success": False, "message": opt_result['message']}
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def optimize_with_black_litterman(self, views=None, confidence=None, tau=0.025):
        """
        Optimize using Black-Litterman model incorporating investor views.
        
        Args:
            views: Dict mapping assets to expected returns (investor views)
            confidence: Dict mapping assets to confidence level (0-1)
            tau: Scalar controlling weight of prior vs. views
            
        Returns:
            dict: Optimization results
        """
        try:
            logger.info("Optimizing portfolio with Black-Litterman model")
            
            # Check if data is loaded
            if self.returns is None or self.cov_matrix is None:
                logger.error("Market data not loaded. Call load_market_data first.")
                return {"success": False, "message": "Market data not loaded"}
            
            # Calculate market-implied returns (reverse optimization)
            # Get current market weights (equal weight for simplicity)
            n_assets = len(self.returns.columns)
            market_weights = np.ones(n_assets) / n_assets
            
            # Calculate implied returns
            implied_returns = self.risk_free_rate + tau * (self.cov_matrix.values @ market_weights)
            implied_returns_series = pd.Series(implied_returns, index=self.cov_matrix.columns)
            
            # Process investor views
            if views is None:
                views = {}  # No views
            
            if confidence is None:
                confidence = {asset: 0.5 for asset in views.keys()}  # Default confidence
            
            # Create view matrix P and view vector Q
            assets = list(self.cov_matrix.columns)
            P = np.zeros((len(views), len(assets)))
            Q = np.zeros(len(views))
            
            # Confidence matrix (diagonal)
            omega = np.zeros((len(views), len(views)))
            
            for i, (asset, view_return) in enumerate(views.items()):
                if asset in assets:
                    asset_idx = assets.index(asset)
                    P[i, asset_idx] = 1.0
                    Q[i] = view_return
                    
                    # Set confidence
                    conf = confidence.get(asset, 0.5)
                    omega[i, i] = (1.0 / conf - 1.0) * (self.cov_matrix.values[asset_idx, asset_idx] * tau)
            
            # If no views provided, use implied returns directly
            if not views:
                bl_returns = implied_returns_series
            else:
                # Apply Black-Litterman formula
                # Compute posterior expected returns
                cov_matrix_scaled = tau * self.cov_matrix.values
                
                # Formula: E[R] = [(tau * Σ)^(-1) + P' * Ω^(-1) * P]^(-1) * [(tau * Σ)^(-1) * π + P' * Ω^(-1) * Q]
                # Where:
                # - E[R] is the posterior expected returns
                # - π is the implied returns
                # - Σ is the covariance matrix
                # - P is the view matrix
                # - Q is the view vector
                # - Ω is the confidence matrix
                
                try:
                    # Calculate components
                    cov_inv = np.linalg.inv(cov_matrix_scaled)
                    if len(views) > 0:
                        omega_inv = np.linalg.inv(omega)
                        term1 = cov_inv + P.T @ omega_inv @ P
                        term2 = cov_inv @ implied_returns + P.T @ omega_inv @ Q
                        posterior_returns = np.linalg.inv(term1) @ term2
                    else:
                        posterior_returns = implied_returns
                    
                    bl_returns = pd.Series(posterior_returns, index=self.cov_matrix.columns)
                    
                except np.linalg.LinAlgError:
                    logger.warning("Linear algebra error, falling back to implied returns")
                    bl_returns = implied_returns_series
            
            # Store original expected returns
            original_expected_returns = self.expected_returns
            
            # Use Black-Litterman returns for optimization
            self.expected_returns = bl_returns
            
            # Optimize portfolio with new expected returns
            result = self.optimize_portfolio(objective='sharpe')
            
            # Restore original expected returns
            self.expected_returns = original_expected_returns
            
            # Return Black-Litterman specific results
            if result['success']:
                result.update({
                    'implied_returns': implied_returns_series,
                    'bl_returns': bl_returns
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def optimize_with_drawdown_constraint(self, max_drawdown=None):
        """
        Optimize portfolio with maximum drawdown constraint.
        
        Args:
            max_drawdown: Maximum allowed drawdown (negative decimal)
            
        Returns:
            dict: Optimization results
        """
        try:
            logger.info("Optimizing portfolio with drawdown constraint")
            
            # Use provided max drawdown or default
            max_dd = max_drawdown if max_drawdown is not None else self.max_drawdown_limit
            
            # Check if data is loaded
            if self.returns is None:
                logger.error("Market data not loaded. Call load_market_data first.")
                return {"success": False, "message": "Market data not loaded"}
            
            # Drawdown constraint function
            def drawdown_constraint(weights):
                # Calculate historical returns with these weights
                historical_returns = self.returns @ weights
                
                # Calculate cumulative returns
                cumulative_returns = (1 + historical_returns).cumprod()
                
                # Calculate drawdown
                historical_drawdown = (cumulative_returns / cumulative_returns.cummax()) - 1
                max_drawdown = historical_drawdown.min()
                
                # Constraint: max_drawdown >= -max_dd
                return max_drawdown + max_dd
            
            # Define constraints
            constraints = [
                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                {'type': 'ineq', 'fun': drawdown_constraint}
            ]
            
            # Run optimization with drawdown constraint
            n_assets = len(self.returns.columns)
            initial_weights = np.ones(n_assets) / n_assets
            bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
            
            opt_result = minimize(
                self._maximize_sharpe,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'disp': False, 'maxiter': 1000}
            )
            
            if opt_result['success']:
                # Get optimized weights
                weights = opt_result['x']
                
                # Ensure no negative weights and normalize
                weights = np.clip(weights, 0, None)
                weights = weights / np.sum(weights)
                
                # Convert to Series
                self.optimal_weights = pd.Series(weights, index=self.returns.columns)
                
                # Calculate portfolio metrics
                self.portfolio_metrics = self._calculate_portfolio_metrics(self.optimal_weights)
                
                logger.info(f"Drawdown-constrained optimization successful, max drawdown: {self.portfolio_metrics['max_drawdown']:.4f}")
                
                return {
                    "success": True,
                    "weights": self.optimal_weights,
                    "metrics": self.portfolio_metrics
                }
            else:
                logger.error(f"Drawdown-constrained optimization failed: {opt_result['message']}")
                return {"success": False, "message": opt_result['message']}
                
        except Exception as e:
            logger.error(f"Error in drawdown-constrained optimization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def plot_efficient_frontier(self, save_path=None, show_assets=True, highlight_portfolio=True):
        """
        Plot the efficient frontier.
        
        Args:
            save_path: Path to save the plot image
            show_assets: Whether to show individual assets on the plot
            highlight_portfolio: Whether to highlight the current portfolio
            
        Returns:
            bool: Success or failure
        """
        try:
            # Generate frontier if not already done
            if self.efficient_frontier is None:
                self.generate_efficient_frontier()
            
            if self.efficient_frontier is None:
                logger.error("Failed to generate efficient frontier")
                return False
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot the efficient frontier
            plt.plot(
                self.efficient_frontier['risk'],
                self.efficient_frontier['return'],
                'b-',
                linewidth=3,
                label='Efficient Frontier'
            )
            
            # Highlight the maximum Sharpe ratio portfolio
            max_sharpe_idx = self.efficient_frontier.get('max_sharpe', False).idxmax() if 'max_sharpe' in self.efficient_frontier.columns else None
            
            if max_sharpe_idx is not None:
                max_sharpe_risk = self.efficient_frontier.loc[max_sharpe_idx, 'risk']
                max_sharpe_return = self.efficient_frontier.loc[max_sharpe_idx, 'return']
                
                plt.plot(
                    max_sharpe_risk,
                    max_sharpe_return,
                    'r*',
                    markersize=15,
                    label='Maximum Sharpe Ratio Portfolio'
                )
            
            # Plot individual assets if requested
            if show_assets:
                for asset in self.returns.columns:
                    # Calculate annualized return and risk for this asset
                    asset_return = self.expected_returns[asset]
                    asset_risk = np.sqrt(self.cov_matrix.loc[asset, asset])
                    
                    plt.plot(
                        asset_risk,
                        asset_return,
                        'go',
                        markersize=8,
                        alpha=0.6
                    )
                    
                    plt.annotate(
                        asset,
                        (asset_risk, asset_return),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8
                    )
            
            # Highlight the current optimal portfolio if requested
            if highlight_portfolio and self.optimal_weights is not None and self.portfolio_metrics is not None:
                portfolio_risk = self.portfolio_metrics['annual_volatility']
                portfolio_return = self.portfolio_metrics['expected_return']
                
                plt.plot(
                    portfolio_risk,
                    portfolio_return,
                    'y^',
                    markersize=12,
                    label='Current Portfolio'
                )
            
            # Plot the capital market line
            min_risk = min(self.efficient_frontier['risk'].min(), 0)
            max_risk = max(self.efficient_frontier['risk'].max(), 0.5)
            
            if max_sharpe_idx is not None:
                # Slope of the capital market line
                sharpe = (max_sharpe_return - self.risk_free_rate) / max_sharpe_risk
                
                # Capital market line points
                cml_x = np.linspace(0, max_risk * 1.2, 100)
                cml_y = self.risk_free_rate + sharpe * cml_x
                
                plt.plot(
                    cml_x,
                    cml_y,
                    'g--',
                    linewidth=2,
                    label='Capital Market Line'
                )
                
                # Plot risk-free rate
                plt.plot(
                    0,
                    self.risk_free_rate,
                    'k*',
                    markersize=10,
                    label=f'Risk-Free Rate ({self.risk_free_rate:.2%})'
                )
            
            # Set labels and title
            plt.xlabel('Annualized Risk (Standard Deviation)', fontsize=12)
            plt.ylabel('Annualized Expected Return', fontsize=12)
            plt.title('Portfolio Efficient Frontier', fontsize=14)
            
            # Format axes as percentages
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            
            # Add grid and legend
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=10)
            
            # Set axis limits
            plt.xlim(min_risk, max_risk * 1.1)
            plt.ylim(self.risk_free_rate * 0.5, self.efficient_frontier['return'].max() * 1.1)
            
            # Add plot description
            description = (
                f"Efficient Frontier Analysis\n"
                f"Number of Assets: {len(self.returns.columns)}\n"
                f"Risk-Free Rate: {self.risk_free_rate:.2%}\n"
            )
            
            if self.portfolio_metrics:
                description += (
                    f"Current Portfolio:\n"
                    f"  Expected Return: {self.portfolio_metrics['expected_return']:.2%}\n"
                    f"  Sharpe Ratio: {self.portfolio_metrics['sharpe_ratio']:.2f}\n"
                    f"  Max Drawdown: {self.portfolio_metrics['max_drawdown']:.2%}\n"
                )
            
            plt.figtext(0.02, 0.02, description, fontsize=9, ha='left')
            
            # Save or show plot
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.tight_layout()
                plt.show()
            
            return True
            
        except Exception as e:
            logger.error(f"Error plotting efficient frontier: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def plot_portfolio_weights(self, save_path=None):
        """
        Plot the portfolio weights as a horizontal bar chart.
        
        Args:
            save_path: Path to save the plot image
            
        Returns:
            bool: Success or failure
        """
        try:
            if self.optimal_weights is None:
                logger.error("No portfolio weights available")
                return False
            
            # Filter out tiny positions for clarity
            filtered_weights = self.optimal_weights[self.optimal_weights >= 0.005]
            
            # Sort by weight (descending)
            sorted_weights = filtered_weights.sort_values(ascending=True)
            
            # Create figure
            plt.figure(figsize=(10, max(6, len(sorted_weights) * 0.4)))
            
            # Plot horizontal bar chart
            bars = plt.barh(
                sorted_weights.index,
                sorted_weights.values,
                color='skyblue',
                edgecolor='navy',
                alpha=0.7
            )
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(
                    width + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f'{width:.1%}',
                    va='center'
                )
            
            # Add sector color coding if available
            if self.sector_map:
                sectors = {}
                for asset, sector in self.sector_map.items():
                    if asset in sorted_weights.index:
                        if sector not in sectors:
                            sectors[sector] = []
                        sectors[sector].append(asset)
                
                # Color bars by sector
                if sectors:
                    cmap = plt.cm.get_cmap('tab10', len(sectors))
                    sector_colors = {sector: cmap(i) for i, sector in enumerate(sectors.keys())}
                    
                    for i, asset in enumerate(sorted_weights.index):
                        if asset in self.sector_map:
                            sector = self.sector_map[asset]
                            bars[i].set_color(sector_colors[sector])
                    
                    # Add legend
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor=color, edgecolor='navy', alpha=0.7, label=sector)
                        for sector, color in sector_colors.items()
                    ]
                    plt.legend(handles=legend_elements, loc='lower right')
            
            # Set labels and title
            plt.xlabel('Portfolio Weight', fontsize=12)
            plt.ylabel('Asset', fontsize=12)
            plt.title('Portfolio Allocation', fontsize=14)
            
            # Format x-axis as percentages
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            
            # Add portfolio metrics text
            if self.portfolio_metrics:
                metrics_text = (
                    f"Expected Return: {self.portfolio_metrics['expected_return']:.2%}\n"
                    f"Volatility: {self.portfolio_metrics['annual_volatility']:.2%}\n"
                    f"Sharpe Ratio: {self.portfolio_metrics['sharpe_ratio']:.2f}\n"
                    f"Max Drawdown: {self.portfolio_metrics['max_drawdown']:.2%}\n"
                    f"Value at Risk (95%): {self.portfolio_metrics['var_95']:.2%}\n"
                )
                plt.figtext(0.02, 0.02, metrics_text, fontsize=9, ha='left')
            
            # Save or show plot
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.tight_layout()
                plt.show()
            
            return True
            
        except Exception as e:
            logger.error(f"Error plotting portfolio weights: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def plot_risk_contribution(self, save_path=None):
        """
        Plot the risk contribution of each asset in the portfolio.
        
        Args:
            save_path: Path to save the plot image
            
        Returns:
            bool: Success or failure
        """
        try:
            if self.portfolio_metrics is None or 'risk_contribution' not in self.portfolio_metrics:
                logger.error("No risk contribution data available")
                return False
            
            # Get risk contribution data
            risk_contrib = pd.Series(self.portfolio_metrics['risk_contribution'])
            
            # Filter and sort
            filtered_risk = risk_contrib[risk_contrib >= 0.01]  # Filter out tiny contributions
            sorted_risk = filtered_risk.sort_values(ascending=True)
            
            # Create figure
            plt.figure(figsize=(10, max(6, len(sorted_risk) * 0.4)))
            
            # Plot horizontal bar chart
            bars = plt.barh(
                sorted_risk.index,
                sorted_risk.values,
                color='salmon',
                edgecolor='darkred',
                alpha=0.7
            )
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(
                    width + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f'{width:.1%}',
                    va='center'
                )
            
            # Set labels and title
            plt.xlabel('Risk Contribution', fontsize=12)
            plt.ylabel('Asset', fontsize=12)
            plt.title('Portfolio Risk Contribution', fontsize=14)
            
            # Format x-axis as percentages
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            
            # Add comparison with weight contribution
            if self.optimal_weights is not None:
                weight_risk_text = "Asset Weight vs Risk Contribution:\n"
                for asset in sorted_risk.index[:10]:  # Top 10 risk contributors
                    weight = self.optimal_weights.get(asset, 0) * 100
                    risk = sorted_risk.get(asset, 0) * 100
                    ratio = risk / weight if weight > 0 else 0
                    weight_risk_text += f"{asset}: {weight:.1f}% weight, {risk:.1f}% risk, {ratio:.2f}x ratio\n"
                
                plt.figtext(0.02, 0.02, weight_risk_text, fontsize=9, ha='left')
            
            # Save or show plot
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.tight_layout()
                plt.show()
            
            return True
            
        except Exception as e:
            logger.error(f"Error plotting risk contribution: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def calculate_rebalancing_trades(self, current_weights, cash_injection=0):
        """
        Calculate trades needed to rebalance the portfolio.
        
        Args:
            current_weights: Current portfolio weights or position values
            cash_injection: Additional cash to invest (positive) or withdraw (negative)
            
        Returns:
            dict: Rebalancing trades and information
        """
        try:
            if self.optimal_weights is None:
                logger.error("No target portfolio weights available")
                return {"success": False, "message": "No target weights available"}
            
            # Convert inputs to Series if they're not already
            if not isinstance(current_weights, pd.Series):
                current_weights = pd.Series(current_weights)
            
            # Calculate portfolio value and adjust for cash
            portfolio_value = current_weights.sum()
            new_portfolio_value = portfolio_value + cash_injection
            
            if new_portfolio_value <= 0:
                return {"success": False, "message": "Invalid portfolio value after cash adjustment"}
            
            # Get target weights for all assets in either current or target
            all_assets = set(current_weights.index) | set(self.optimal_weights.index)
            
            # Initialize Series with zeros for all assets
            current_full = pd.Series(0, index=all_assets)
            target_full = pd.Series(0, index=all_assets)
            
            # Fill in actual values
            for asset in all_assets:
                if asset in current_weights:
                    current_full[asset] = current_weights[asset]
                if asset in self.optimal_weights:
                    target_full[asset] = self.optimal_weights[asset] * new_portfolio_value
            
            # Calculate trades
            trades = target_full - current_full
            
            # Verify that trades sum to cash_injection (within rounding error)
            trade_sum = trades.sum()
            if abs(trade_sum - cash_injection) > 1e-6 * new_portfolio_value:
                logger.warning(f"Trade sum {trade_sum} doesn't match cash injection {cash_injection}")
                # Adjust largest trade to ensure cash balance
                adjustment = cash_injection - trade_sum
                largest_trade_idx = trades.abs().idxmax()
                trades[largest_trade_idx] += adjustment
            
            # Calculate percentage changes
            pct_changes = {}
            for asset in all_assets:
                if current_full[asset] > 0:
                    pct_changes[asset] = trades[asset] / current_full[asset]
                else:
                    pct_changes[asset] = np.inf if trades[asset] > 0 else -np.inf
            
            # Create results
            results = {
                "success": True,
                "trades": trades,
                "pct_changes": pd.Series(pct_changes),
                "current_weights": current_full / portfolio_value if portfolio_value > 0 else pd.Series(0, index=all_assets),
                "target_weights": target_full / new_portfolio_value,
                "original_portfolio_value": portfolio_value,
                "new_portfolio_value": new_portfolio_value,
                "cash_injection": cash_injection
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating rebalancing trades: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def get_portfolio_allocation(self, total_investment, ml_weights=None):
        """
        Get the portfolio allocation for a given investment amount.
        
        Args:
            total_investment: Total investment amount
            ml_weights: Optional ML-generated weights to adjust allocation
            
        Returns:
            dict: Portfolio allocation
        """
        try:
            if self.optimal_weights is None:
                logger.error("No portfolio weights available")
                return {}
            
            # Basic allocation based on optimal weights
            allocation = {}
            
            # Apply ML weights if provided
            if ml_weights is not None:
                # Combine optimal weights with ML weights
                adjusted_weights = {}
                
                for asset, weight in self.optimal_weights.items():
                    if asset in ml_weights:
                        # Use ML weight to adjust the optimal weight
                        ml_weight = ml_weights[asset]
                        # Scale the adjustment to avoid extreme changes
                        adjustment_factor = 1.0 + (ml_weight / 1000)
                        adjusted_weights[asset] = weight * adjustment_factor
                    else:
                        adjusted_weights[asset] = weight
                
                # Normalize weights
                total = sum(adjusted_weights.values())
                normalized_weights = {asset: w/total for asset, w in adjusted_weights.items()}
                
                # Calculate allocation
                for asset, weight in normalized_weights.items():
                    allocation[asset] = {
                        "weight": weight,
                        "amount": weight * total_investment,
                        "optimal_weight": self.optimal_weights.get(asset, 0),
                        "ml_adjustment": ml_weights.get(asset, 0) if ml_weights else 0
                    }
            else:
                # Use optimal weights directly
                for asset, weight in self.optimal_weights.items():
                    allocation[asset] = {
                        "weight": weight,
                        "amount": weight * total_investment
                    }
            
            # Add portfolio metrics
            allocation["portfolio_metrics"] = self.portfolio_metrics
            
            return allocation
            
        except Exception as e:
            logger.error(f"Error calculating portfolio allocation: {e}")
            return {}

# Demo usage
if __name__ == "__main__":
    import yfinance as yf
    
    # Download sample price data
    print("Downloading sample price data...")
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'PG']
    
    price_data = yf.download(tickers, period="2y")['Adj Close']
    
    # Create sector mapping
    sectors = {
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'AMZN': 'Consumer Discretionary',
        'GOOGL': 'Communication Services',
        'META': 'Communication Services',
        'TSLA': 'Consumer Discretionary',
        'NVDA': 'Technology',
        'JPM': 'Financials',
        'JNJ': 'Healthcare',
        'PG': 'Consumer Staples'
    }
    
    # Initialize portfolio optimizer
    optimizer = PortfolioOptimizer()
    
    # Load market data
    print("Loading market data...")
    optimizer.load_market_data(price_data, sectors)
    
    # Generate efficient frontier
    print("Generating efficient frontier...")
    frontier = optimizer.generate_efficient_frontier()
    
    # Optimize portfolio (Sharpe ratio)
    print("\nOptimizing portfolio (Sharpe ratio)...")
    result = optimizer.optimize_portfolio(objective='sharpe')
    print("Optimal weights:")
    print(optimizer.optimal_weights)
    print("\nPortfolio metrics:")
    for key, value in optimizer.portfolio_metrics.items():
        if key not in ['composition', 'risk_contribution', 'sector_allocation']:
            print(f"{key}: {value}")
    
    # Optimize with Kelly criterion
    print("\nOptimizing with Kelly criterion...")
    kelly_result = optimizer.optimize_with_kelly()
    
    # Optimize with risk parity
    print("\nOptimizing with risk parity...")
    rp_result = optimizer.optimize_with_risk_parity()
    
    # Optimize with drawdown constraint
    print("\nOptimizing with drawdown constraint...")
    dd_result = optimizer.optimize_with_drawdown_constraint(max_drawdown=-0.1)
    
    # Plot efficient frontier
    print("\nPlotting efficient frontier...")
    optimizer.plot_efficient_frontier()
    
    # Plot portfolio weights
    print("\nPlotting portfolio weights...")
    optimizer.plot_portfolio_weights()
    
    # Plot risk contribution
    print("\nPlotting risk contribution...")
    optimizer.plot_risk_contribution()
    
    # Calculate rebalancing trades
    print("\nCalculating rebalancing trades...")
    # Simulate current positions (random proportions)
    np.random.seed(42)
    current_positions = np.random.random(len(tickers))
    current_positions = current_positions / sum(current_positions) * 10000  # $10,000 portfolio
    current_weights = pd.Series(current_positions, index=tickers)
    
    trades = optimizer.calculate_rebalancing_trades(current_weights, cash_injection=2000)
    
    print("Rebalancing trades:")
    for asset, trade in trades['trades'].items():
        if abs(trade) > 1:  # Only show non-trivial trades
            print(f"{asset}: {'Buy' if trade > 0 else 'Sell'} ${abs(trade):.2f}")