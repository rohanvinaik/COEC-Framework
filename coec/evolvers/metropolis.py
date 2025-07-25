"""
Metropolis-Hastings evolution operator for stochastic sampling
"""
import numpy as np
from typing import Optional, Callable, Dict, Any

from ..core.evolution import EvolutionOperator
from ..core.substrate import Substrate
from ..substrates.euclidean import EuclideanSubstrate


class MetropolisHastingsEvolver(EvolutionOperator):
    """
    Generic Metropolis-Hastings sampler (§9.1 – Algorithm 9.1).
    
    This evolver implements the Metropolis-Hastings algorithm for sampling
    from the joint distribution defined by energy and constraints. The
    acceptance probability is:
    
    P(accept) = min(1, exp(-ΔE/T + Σ p_i log(c_i(x')/c_i(x))))
    
    Where:
    - ΔE: energy difference
    - T: temperature
    - p_i: precision weight for constraint i
    - c_i(x): constraint satisfaction at state x
    
    This allows sampling from complex, multi-modal distributions while
    respecting both energetic and constraint preferences.
    """
    
    def __init__(
        self,
        substrate: Substrate,
        constraints,
        energy,
        temperature: float = 1.0,
        proposal_std: float = 0.1,
        adaptive_proposal: bool = False,
        proposal_fn: Optional[Callable] = None,
        rng=None,
        name=None
    ):
        """
        Initialize Metropolis-Hastings evolver.
        
        Args:
            substrate: Initial substrate state
            constraints: List of constraints
            energy: Energy landscape
            temperature: Sampling temperature (higher = more exploration)
            proposal_std: Standard deviation for Gaussian proposals
            adaptive_proposal: Whether to adapt proposal distribution
            proposal_fn: Custom proposal function (optional)
            rng: Random number generator
            name: Optional name
        """
        super().__init__(substrate, constraints, energy, rng, name)
        
        self.temperature = temperature
        self.proposal_std = proposal_std
        self.adaptive_proposal = adaptive_proposal
        self.proposal_fn = proposal_fn
        
        # Track acceptance statistics
        self.n_proposed = 0
        self.n_accepted = 0
        self.acceptance_history = []
        
        # For adaptive proposals
        if adaptive_proposal:
            self.proposal_history = []
            self.target_acceptance = 0.234  # Optimal for high dimensions
    
    def step(self, dt: float = 1.0) -> None:
        """
        Perform one Metropolis-Hastings step.
        
        Args:
            dt: Unused for discrete-time algorithm
        """
        # Generate proposal
        current = self.S.clone()
        proposal = self._generate_proposal(current)
        
        # Compute acceptance probability
        log_accept = self._compute_log_acceptance(current, proposal)
        
        # Accept or reject
        self.n_proposed += 1
        if np.log(self.rng.uniform()) < log_accept:
            # Accept proposal
            self.S.state = proposal.state
            self.n_accepted += 1
            accepted = True
        else:
            # Reject proposal (state unchanged)
            accepted = False
        
        # Update statistics
        self.acceptance_history.append(accepted)
        if len(self.acceptance_history) > 1000:
            self.acceptance_history.pop(0)
        
        # Adapt proposal if enabled
        if self.adaptive_proposal:
            self._adapt_proposal()
        
        # Record step
        self.step_count += 1
        self.record_step()
    
    def _generate_proposal(self, current: Substrate) -> Substrate:
        """
        Generate proposal state from current state.
        
        Args:
            current: Current substrate state
            
        Returns:
            Proposed substrate state
        """
        if self.proposal_fn is not None:
            # Use custom proposal function
            return self.proposal_fn(current, self.rng)
        
        # Default: Gaussian perturbation for Euclidean substrates
        if isinstance(current, EuclideanSubstrate):
            perturbation = self.rng.normal(
                scale=self.proposal_std,
                size=current.state.shape
            )
            new_state = current.state + perturbation
            return EuclideanSubstrate(new_state)
        else:
            raise NotImplementedError(
                f"No default proposal for {type(current)}. "
                "Please provide a custom proposal_fn."
            )
    
    def _compute_log_acceptance(self, current: Substrate, proposal: Substrate) -> float:
        """
        Compute log acceptance probability.
        
        Args:
            current: Current state
            proposal: Proposed state
            
        Returns:
            Log of acceptance probability
        """
        # Energy difference
        current_energy = self.E.energy(current)
        proposal_energy = self.E.energy(proposal)
        delta_E = proposal_energy - current_energy
        
        # Constraint satisfaction ratio (in log space for stability)
        log_constraint_ratio = 0.0
        for c in self.C:
            current_sat = c.evaluate(current)
            proposal_sat = c.evaluate(proposal)
            
            # Avoid log(0) by adding small epsilon
            eps = 1e-12
            log_ratio = np.log(proposal_sat + eps) - np.log(current_sat + eps)
            log_constraint_ratio += c.precision * log_ratio
        
        # Metropolis-Hastings acceptance
        log_accept = -delta_E / self.temperature + log_constraint_ratio
        
        return log_accept
    
    def _adapt_proposal(self) -> None:
        """
        Adapt proposal distribution based on acceptance rate.
        
        Uses the Robbins-Monro algorithm to achieve target acceptance rate.
        """
        if len(self.acceptance_history) < 50:
            return  # Need enough history
        
        # Current acceptance rate
        current_rate = np.mean(self.acceptance_history[-50:])
        
        # Adapt step size to achieve target acceptance
        # Increase proposal_std if accepting too much (explore more)
        # Decrease if accepting too little (explore less)
        adaptation_rate = 0.01
        if current_rate > self.target_acceptance:
            self.proposal_std *= (1 + adaptation_rate)
        else:
            self.proposal_std *= (1 - adaptation_rate)
        
        # Keep in reasonable range
        self.proposal_std = np.clip(self.proposal_std, 1e-4, 10.0)
    
    def get_acceptance_rate(self) -> float:
        """
        Get overall acceptance rate.
        
        Returns:
            Fraction of proposals accepted
        """
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed
    
    def get_recent_acceptance_rate(self, window: int = 100) -> float:
        """
        Get acceptance rate over recent steps.
        
        Args:
            window: Number of recent steps to consider
            
        Returns:
            Recent acceptance rate
        """
        if not self.acceptance_history:
            return 0.0
        
        recent = self.acceptance_history[-window:]
        return np.mean(recent)
    
    def set_temperature(self, temperature: float) -> None:
        """
        Update sampling temperature.
        
        Args:
            temperature: New temperature value
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = temperature
    
    def anneal_temperature(self, factor: float = 0.99) -> None:
        """
        Reduce temperature by multiplicative factor (simulated annealing).
        
        Args:
            factor: Temperature reduction factor (should be < 1)
        """
        self.temperature *= factor
    
    def get_sampling_info(self) -> Dict[str, Any]:
        """
        Get information about sampling performance.
        
        Returns:
            Dictionary with sampling metrics
        """
        return {
            'steps': self.step_count,
            'temperature': self.temperature,
            'proposal_std': self.proposal_std,
            'acceptance_rate': self.get_acceptance_rate(),
            'recent_acceptance_rate': self.get_recent_acceptance_rate(),
            'current_energy': self.E.energy(self.S),
            'current_satisfaction': self.compute_total_satisfaction()
        }
    
    def __repr__(self) -> str:
        """String representation with sampling info."""
        return (
            f"MetropolisHastingsEvolver("
            f"T={self.temperature:.3f}, "
            f"accept_rate={self.get_acceptance_rate():.3f}, "
            f"steps={self.step_count})"
        )