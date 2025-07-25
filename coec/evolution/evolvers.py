"""
Evolution operators for COEC systems.

These implement the Φ operator that maps initial states to trajectories.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from ..core.system import EvolutionOperator, Substrate, Constraint, EnergyLandscape


class GradientDescentEvolver(EvolutionOperator):
    """
    Gradient descent evolution operator.
    
    Evolves the system by following gradients of the combined
    energy-constraint landscape.
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def evolve(self,
               substrate: Substrate,
               constraints: List[Constraint],
               energy_landscape: EnergyLandscape,
               steps: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Evolve system using gradient descent.
        """
        trajectory = []
        state = substrate.get_state()
        velocity = np.zeros_like(state)
        
        energy_history = []
        constraint_history = {c.name: [] for c in constraints}
        
        for step in range(steps):
            # Store current state
            trajectory.append(state.copy())
            
            # Compute energy gradient
            energy_grad = energy_landscape.compute_gradient(state)
            
            # Compute constraint gradients (weighted by precision)
            constraint_grad = np.zeros_like(state)
            for constraint in constraints:
                c_grad = constraint.gradient(state)
                constraint_grad += constraint.precision * c_grad
                
                # Track constraint satisfaction
                constraint_history[constraint.name].append(
                    constraint.satisfaction(state)
                )
            
            # Combined gradient
            total_grad = energy_grad + constraint_grad
            
            # Update with momentum
            velocity = self.momentum * velocity - self.learning_rate * total_grad
            state = state + velocity
            
            # Track energy
            energy_history.append(energy_landscape.compute_energy(state))
        
        # Update substrate with final state
        substrate.set_state(state)
        
        metadata = {
            "energy_history": energy_history,
            "constraint_history": constraint_history,
            "final_velocity": velocity
        }
        
        return np.array(trajectory), metadata


class MetropolisHastingsEvolver(EvolutionOperator):
    """
    Metropolis-Hastings Monte Carlo evolution operator.
    
    Implements the transition probability:
    P(ω_a → ω_b) = (1/Z)exp(-(E(ω_b) - E(ω_a))/(k_B T)) ∏ c_i(ω_b)^p_i
    """
    
    def __init__(self, temperature: float = 1.0, step_size: float = 0.1):
        self.temperature = temperature
        self.step_size = step_size
    
    def evolve(self,
               substrate: Substrate,
               constraints: List[Constraint],
               energy_landscape: EnergyLandscape,
               steps: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Evolve system using Metropolis-Hastings algorithm.
        """
        trajectory = []
        state = substrate.get_state()
        
        energy_history = []
        constraint_history = {c.name: [] for c in constraints}
        acceptance_history = []
        
        current_energy = energy_landscape.compute_energy(state)
        
        for step in range(steps):
            # Store current state
            trajectory.append(state.copy())
            
            # Propose new state
            proposal = state + self.step_size * np.random.randn(*state.shape)
            
            # Compute energies
            proposal_energy = energy_landscape.compute_energy(proposal)
            energy_diff = proposal_energy - current_energy
            
            # Compute constraint satisfaction product
            current_constraint_prod = 1.0
            proposal_constraint_prod = 1.0
            
            for constraint in constraints:
                current_sat = constraint.satisfaction(state)
                proposal_sat = constraint.satisfaction(proposal)
                
                current_constraint_prod *= current_sat ** constraint.precision
                proposal_constraint_prod *= proposal_sat ** constraint.precision
                
                # Track satisfaction
                constraint_history[constraint.name].append(current_sat)
            
            # Acceptance probability
            if proposal_constraint_prod > 0:
                constraint_ratio = proposal_constraint_prod / max(current_constraint_prod, 1e-10)
                acceptance_prob = min(1.0, 
                    constraint_ratio * np.exp(-energy_diff / self.temperature)
                )
            else:
                acceptance_prob = 0.0
            
            # Accept or reject
            if np.random.rand() < acceptance_prob:
                state = proposal
                current_energy = proposal_energy
                acceptance_history.append(1)
            else:
                acceptance_history.append(0)
            
            energy_history.append(current_energy)
        
        # Update substrate with final state
        substrate.set_state(state)
        
        metadata = {
            "energy_history": energy_history,
            "constraint_history": constraint_history,
            "acceptance_rate": np.mean(acceptance_history),
            "acceptance_history": acceptance_history
        }
        
        return np.array(trajectory), metadata


class AdaptiveEvolver(EvolutionOperator):
    """
    Adaptive evolution operator that modifies constraints during evolution.
    
    This implements AP-COEC (Adaptive-Plastic) systems where
    dC/dt = g(S, C, E).
    """
    
    def __init__(self, 
                 base_evolver: EvolutionOperator,
                 adaptation_rate: float = 0.01,
                 adaptation_threshold: float = 0.5):
        self.base_evolver = base_evolver
        self.adaptation_rate = adaptation_rate
        self.adaptation_threshold = adaptation_threshold
    
    def evolve(self,
               substrate: Substrate,
               constraints: List[Constraint],
               energy_landscape: EnergyLandscape,
               steps: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Evolve system with adaptive constraint modification.
        """
        trajectory = []
        precision_history = {c.name: [] for c in constraints}
        
        # Divide evolution into epochs
        epochs = 10
        steps_per_epoch = steps // epochs
        
        for epoch in range(epochs):
            # Run base evolution for one epoch
            epoch_traj, epoch_meta = self.base_evolver.evolve(
                substrate, constraints, energy_landscape, steps_per_epoch
            )
            
            trajectory.extend(epoch_traj)
            
            # Adapt constraints based on performance
            for constraint in constraints:
                # Get average satisfaction in this epoch
                satisfaction_history = epoch_meta["constraint_history"][constraint.name]
                avg_satisfaction = np.mean(satisfaction_history)
                
                # Record current precision
                precision_history[constraint.name].append(constraint.precision)
                
                # Adapt precision based on satisfaction
                if avg_satisfaction < self.adaptation_threshold:
                    # Increase precision for poorly satisfied constraints
                    constraint.precision *= (1 + self.adaptation_rate)
                else:
                    # Decrease precision for well-satisfied constraints
                    constraint.precision *= (1 - self.adaptation_rate)
                
                # Keep precision in reasonable bounds
                constraint.precision = np.clip(constraint.precision, 0.1, 10.0)
        
        metadata = {
            "precision_history": precision_history,
            "final_precisions": {c.name: c.precision for c in constraints}
        }
        
        return np.array(trajectory), metadata
