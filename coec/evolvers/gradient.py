"""
Gradient descent evolution operator with constraint awareness
"""
import numpy as np
from typing import Optional, Dict, Any

from ..core.evolution import EvolutionOperator
from ..core.substrate import Substrate
from ..substrates.euclidean import EuclideanSubstrate


class GradientDescentEvolver(EvolutionOperator):
    """
    Constraint-aware gradient descent (§9.2 – Algorithm 9.2).
    
    This evolver implements gradient descent that balances energy
    minimization with constraint satisfaction. The update rule is:
    
    x_{t+1} = x_t - lr_E * ∇E + lr_C * Σ p_i ∇c_i
    
    Where:
    - lr_E: learning rate for energy descent
    - lr_C: learning rate for constraint ascent
    - p_i: precision weight for constraint i
    
    The sign difference ensures we descend energy while ascending
    constraint satisfaction.
    """
    
    def __init__(
        self,
        substrate: Substrate,
        constraints,
        energy,
        lr_E: float = 0.01,
        lr_C: float = 0.01,
        momentum: float = 0.0,
        adaptive_lr: bool = False,
        clip_grad: Optional[float] = None,
        rng=None,
        name=None
    ):
        """
        Initialize gradient descent evolver.
        
        Args:
            substrate: Initial substrate state
            constraints: List of constraints
            energy: Energy landscape
            lr_E: Learning rate for energy gradient
            lr_C: Learning rate for constraint gradients
            momentum: Momentum coefficient (0 = no momentum)
            adaptive_lr: Whether to use adaptive learning rates
            clip_grad: Maximum gradient norm (None = no clipping)
            rng: Random number generator (unused but kept for compatibility)
            name: Optional name
        """
        super().__init__(substrate, constraints, energy, rng, name)
        
        self.lr_E = lr_E
        self.lr_C = lr_C
        self.momentum = momentum
        self.adaptive_lr = adaptive_lr
        self.clip_grad = clip_grad
        
        # Initialize momentum buffer
        if momentum > 0:
            if isinstance(substrate, EuclideanSubstrate):
                self.velocity = np.zeros_like(substrate.state)
            else:
                self.velocity = None
        
        # Adaptive learning rate history
        if adaptive_lr:
            self.lr_history: Dict[str, list] = {
                'energy_reduction': [],
                'constraint_improvement': []
            }
    
    def step(self, dt: float = 1.0) -> None:
        """
        Perform one gradient descent step.
        
        Args:
            dt: Time step multiplier
            
        Raises:
            TypeError: If substrate is not EuclideanSubstrate
        """
        if not isinstance(self.S, EuclideanSubstrate):
            raise TypeError("GradientDescentEvolver requires EuclideanSubstrate")
        
        # Store old values for adaptive learning rate
        if self.adaptive_lr:
            old_energy = self.E.energy(self.S)
            old_satisfaction = self.compute_total_satisfaction()
        
        # Compute energy gradient
        grad_E = self.E.grad(self.S)
        if grad_E is None:
            # Numerical gradient if analytical not available
            grad_E = self._numerical_gradient_energy()
        
        # Compute constraint gradients
        grad_C = np.zeros_like(self.S.state)
        for c in self.C:
            c_grad = c.grad(self.S)
            if c_grad is None:
                # Numerical gradient if analytical not available
                c_grad = self._numerical_gradient_constraint(c)
            grad_C += c.precision * c_grad
        
        # Combine gradients (note sign difference)
        total_grad = self.lr_E * grad_E - self.lr_C * grad_C
        
        # Clip gradient if requested
        if self.clip_grad is not None:
            grad_norm = np.linalg.norm(total_grad)
            if grad_norm > self.clip_grad:
                total_grad = total_grad * (self.clip_grad / grad_norm)
        
        # Apply momentum if enabled
        if self.momentum > 0 and self.velocity is not None:
            self.velocity = self.momentum * self.velocity + (1 - self.momentum) * total_grad
            update = dt * self.velocity
        else:
            update = dt * total_grad
        
        # Update substrate
        self.S.state -= update
        
        # Adapt learning rates if enabled
        if self.adaptive_lr:
            self._adapt_learning_rates(old_energy, old_satisfaction)
        
        # Record step
        self.step_count += 1
        self.record_step(np.linalg.norm(update))
    
    def _numerical_gradient_energy(self, epsilon: float = 1e-6) -> np.ndarray:
        """
        Compute energy gradient using finite differences.
        
        Args:
            epsilon: Finite difference step size
            
        Returns:
            Numerical gradient
        """
        grad = np.zeros_like(self.S.state)
        
        for i in range(len(self.S.state)):
            # Forward difference
            self.S.state[i] += epsilon
            f_plus = self.E.energy(self.S)
            self.S.state[i] -= epsilon
            
            # Backward difference
            self.S.state[i] -= epsilon
            f_minus = self.E.energy(self.S)
            self.S.state[i] += epsilon
            
            # Central difference
            grad[i] = (f_plus - f_minus) / (2 * epsilon)
        
        return grad
    
    def _numerical_gradient_constraint(self, constraint, epsilon: float = 1e-6) -> np.ndarray:
        """
        Compute constraint gradient using finite differences.
        
        Args:
            constraint: Constraint to differentiate
            epsilon: Finite difference step size
            
        Returns:
            Numerical gradient
        """
        grad = np.zeros_like(self.S.state)
        
        for i in range(len(self.S.state)):
            # Forward difference
            self.S.state[i] += epsilon
            f_plus = constraint.evaluate(self.S)
            self.S.state[i] -= epsilon
            
            # Backward difference
            self.S.state[i] -= epsilon
            f_minus = constraint.evaluate(self.S)
            self.S.state[i] += epsilon
            
            # Central difference
            grad[i] = (f_plus - f_minus) / (2 * epsilon)
        
        return grad
    
    def _adapt_learning_rates(self, old_energy: float, old_satisfaction: float) -> None:
        """
        Adapt learning rates based on progress.
        
        Args:
            old_energy: Energy before step
            old_satisfaction: Total satisfaction before step
        """
        new_energy = self.E.energy(self.S)
        new_satisfaction = self.compute_total_satisfaction()
        
        # Track improvements
        energy_improved = new_energy < old_energy
        constraints_improved = new_satisfaction > old_satisfaction
        
        self.lr_history['energy_reduction'].append(energy_improved)
        self.lr_history['constraint_improvement'].append(constraints_improved)
        
        # Adapt based on recent history (last 10 steps)
        if len(self.lr_history['energy_reduction']) >= 10:
            recent_energy = self.lr_history['energy_reduction'][-10:]
            recent_constraint = self.lr_history['constraint_improvement'][-10:]
            
            # Increase lr if consistent improvement, decrease if not
            energy_success_rate = sum(recent_energy) / len(recent_energy)
            constraint_success_rate = sum(recent_constraint) / len(recent_constraint)
            
            if energy_success_rate > 0.8:
                self.lr_E *= 1.05  # Increase by 5%
            elif energy_success_rate < 0.3:
                self.lr_E *= 0.95  # Decrease by 5%
            
            if constraint_success_rate > 0.8:
                self.lr_C *= 1.05
            elif constraint_success_rate < 0.3:
                self.lr_C *= 0.95
            
            # Keep learning rates in reasonable range
            self.lr_E = np.clip(self.lr_E, 1e-6, 1.0)
            self.lr_C = np.clip(self.lr_C, 1e-6, 1.0)
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """
        Get information about convergence status.
        
        Returns:
            Dictionary with convergence metrics
        """
        info = {
            'steps': self.step_count,
            'current_energy': self.E.energy(self.S),
            'current_satisfaction': self.compute_total_satisfaction(),
            'learning_rates': {'energy': self.lr_E, 'constraint': self.lr_C}
        }
        
        if self.history['step_sizes']:
            recent_steps = self.history['step_sizes'][-10:]
            info['avg_step_size'] = np.mean(recent_steps)
            info['step_size_trend'] = recent_steps[-1] / recent_steps[0] if recent_steps[0] > 0 else 1.0
        
        # Check gradient magnitudes
        grad_E = self.E.grad(self.S)
        if grad_E is not None:
            info['energy_grad_norm'] = np.linalg.norm(grad_E)
        
        return info