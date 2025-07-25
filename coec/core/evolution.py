"""
Evolution operator base class - Updates substrate over time
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Optional, Dict, Any
import numpy as np

from .substrate import Substrate
from .constraint import Constraint
from .energy import EnergyLandscape


class EvolutionOperator(ABC):
    """
    Φ – Updates substrate over Δt subject to constraints & energy.
    
    The evolution operator defines how the system progresses through its
    state space, balancing energy minimization with constraint satisfaction.
    Different operators implement different dynamics (gradient descent,
    stochastic sampling, quantum evolution, etc.).
    
    References:
        - §2.5: Computation as physical evolution
        - §7.3: Evolution operator design patterns
    """
    
    def __init__(
        self,
        substrate: Substrate,
        constraints: Sequence[Constraint],
        energy: EnergyLandscape,
        rng: Optional[np.random.Generator] = None,
        name: Optional[str] = None
    ):
        """
        Initialize evolution operator.
        
        Args:
            substrate: Initial substrate state
            constraints: List of constraints to satisfy
            energy: Energy landscape governing dynamics
            rng: Random number generator for stochastic methods
            name: Optional name for the operator
        """
        self.S = substrate
        self.C = list(constraints)
        self.E = energy
        self.rng = rng or np.random.default_rng()
        self.name = name or self.__class__.__name__
        
        # Track evolution history
        self.step_count = 0
        self.history: Dict[str, list] = {
            'energy': [],
            'constraint_satisfaction': [],
            'step_sizes': []
        }
    
    @abstractmethod
    def step(self, dt: float = 1.0) -> None:
        """
        Perform one evolution step, updating substrate in-place.
        
        This is the core method that each evolution operator must implement.
        It should modify self.S according to the specific dynamics of the
        operator (e.g., gradient descent, Metropolis-Hastings, etc.).
        
        Args:
            dt: Time step size (interpretation depends on operator)
        """
        pass
    
    def evolve(self, steps: int, dt: float = 1.0) -> None:
        """
        Perform multiple evolution steps.
        
        Args:
            steps: Number of steps to perform
            dt: Time step size for each step
        """
        for _ in range(steps):
            self.step(dt)
    
    def compute_total_satisfaction(self) -> float:
        """
        Compute weighted sum of all constraint satisfactions.
        
        Returns:
            Sum of precision-weighted constraint satisfactions
        """
        return sum(c.weighted_satisfaction(self.S) for c in self.C)
    
    def compute_constraint_violations(self, threshold: float = 0.9) -> list[str]:
        """
        Identify constraints that are not satisfied above threshold.
        
        Args:
            threshold: Satisfaction threshold
            
        Returns:
            List of unsatisfied constraint names
        """
        violations = []
        for c in self.C:
            if not c.is_satisfied(self.S, threshold):
                violations.append(c.name)
        return violations
    
    def record_step(self, step_size: Optional[float] = None) -> None:
        """
        Record current state metrics for history tracking.
        
        Args:
            step_size: Optional step size to record
        """
        self.history['energy'].append(self.E.energy(self.S))
        self.history['constraint_satisfaction'].append(
            self.compute_total_satisfaction()
        )
        if step_size is not None:
            self.history['step_sizes'].append(step_size)
    
    def reset_history(self) -> None:
        """Clear evolution history."""
        self.step_count = 0
        for key in self.history:
            self.history[key].clear()
    
    def __repr__(self) -> str:
        """String representation of the evolution operator."""
        n_constraints = len(self.C)
        return f"{self.name}(constraints={n_constraints}, steps={self.step_count})"