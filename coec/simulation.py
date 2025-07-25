"""
High-level simulation API for COEC framework
"""
from typing import Any, Sequence, Type, Optional, Dict, Callable
import numpy as np
from tqdm import tqdm

from .core.substrate import Substrate
from .core.constraint import Constraint
from .core.energy import EnergyLandscape
from .core.evolution import EvolutionOperator
from .core.residual import ResidualFunction
from .evolvers.gradient import GradientDescentEvolver


def run_simulation(
    substrate: Substrate,
    constraints: Sequence[Constraint],
    energy: EnergyLandscape,
    evolver_cls: Type[EvolutionOperator] = GradientDescentEvolver,
    n_steps: int = 1000,
    dt: float = 1.0,
    residual: Optional[ResidualFunction] = None,
    evolver_kwargs: Optional[Dict[str, Any]] = None,
    callbacks: Optional[Sequence[Callable]] = None,
    verbose: bool = False,
    record_trajectory: bool = False
) -> Any:
    """
    High-level helper to iterate Φ and return residual output (R).
    
    This function provides a convenient interface for running COEC
    simulations with various evolution operators and analysis options.
    
    Args:
        substrate: Initial substrate state
        constraints: List of constraints to satisfy
        energy: Energy landscape governing dynamics
        evolver_cls: Evolution operator class to use
        n_steps: Number of evolution steps
        dt: Time step size
        residual: Function to extract computational output
        evolver_kwargs: Additional arguments for evolver initialization
        callbacks: List of functions called after each step
        verbose: Whether to show progress bar
        record_trajectory: Whether to record full state trajectory
        
    Returns:
        If residual is provided: residual(final_substrate)
        Otherwise: final substrate state
        
    Example:
        >>> # Minimize quadratic energy with linear constraints
        >>> substrate = EuclideanSubstrate(state=np.random.randn(4))
        >>> energy = QuadraticEnergy(Q=np.eye(4))
        >>> constraints = [
        ...     LinearConstraint(w=np.array([1, 0, 0, 0]), b=0.5),
        ...     LinearConstraint(w=np.array([0, 1, 0, 0]), b=-0.2)
        ... ]
        >>> final_state = run_simulation(
        ...     substrate, constraints, energy, n_steps=1000
        ... )
    """
    # Initialize evolver
    evolver_kwargs = evolver_kwargs or {}
    evolver = evolver_cls(
        substrate=substrate,
        constraints=constraints,
        energy=energy,
        **evolver_kwargs
    )
    
    # Setup trajectory recording if requested
    trajectory = []
    if record_trajectory:
        trajectory.append(substrate.clone())
    
    # Setup progress bar if verbose
    iterator = range(n_steps)
    if verbose:
        iterator = tqdm(iterator, desc="COEC Simulation")
    
    # Main evolution loop
    for step in iterator:
        # Perform evolution step
        evolver.step(dt)
        
        # Record trajectory
        if record_trajectory:
            trajectory.append(evolver.S.clone())
        
        # Call callbacks
        if callbacks:
            for callback in callbacks:
                callback(evolver, step)
        
        # Update progress bar
        if verbose:
            energy_val = evolver.E.energy(evolver.S)
            satisfaction = evolver.compute_total_satisfaction()
            iterator.set_postfix({
                'E': f'{energy_val:.3f}',
                'C': f'{satisfaction:.3f}'
            })
    
    # Extract result
    if residual:
        result = residual(evolver.S)
    else:
        result = evolver.S
    
    # Attach trajectory if recorded
    if record_trajectory:
        if hasattr(result, '__dict__'):
            result.trajectory = trajectory
        else:
            # Return tuple if result doesn't support attributes
            return result, trajectory
    
    return result


class SimulationResult:
    """
    Container for simulation results with analysis methods.
    """
    
    def __init__(
        self,
        final_substrate: Substrate,
        evolver: EvolutionOperator,
        trajectory: Optional[Sequence[Substrate]] = None
    ):
        """
        Initialize simulation result.
        
        Args:
            final_substrate: Final substrate state
            evolver: Evolution operator used
            trajectory: Optional sequence of substrate states
        """
        self.final_substrate = final_substrate
        self.evolver = evolver
        self.trajectory = trajectory
    
    @property
    def final_energy(self) -> float:
        """Get final energy value."""
        return self.evolver.E.energy(self.final_substrate)
    
    @property
    def final_satisfaction(self) -> float:
        """Get final total constraint satisfaction."""
        return self.evolver.compute_total_satisfaction()
    
    def get_constraint_satisfactions(self) -> Dict[str, float]:
        """
        Get individual constraint satisfaction values.
        
        Returns:
            Dictionary mapping constraint names to satisfaction values
        """
        return {
            c.name: c.evaluate(self.final_substrate)
            for c in self.evolver.C
        }
    
    def get_violations(self, threshold: float = 0.9) -> list[str]:
        """
        Get list of violated constraints.
        
        Args:
            threshold: Satisfaction threshold
            
        Returns:
            List of violated constraint names
        """
        return self.evolver.compute_constraint_violations(threshold)
    
    def get_history(self) -> Dict[str, Any]:
        """Get evolution history from evolver."""
        return self.evolver.history
    
    def __repr__(self) -> str:
        """String representation of results."""
        return (
            f"SimulationResult("
            f"steps={self.evolver.step_count}, "
            f"energy={self.final_energy:.3f}, "
            f"satisfaction={self.final_satisfaction:.3f})"
        )


def run_simulation_with_analysis(
    substrate: Substrate,
    constraints: Sequence[Constraint],
    energy: EnergyLandscape,
    evolver_cls: Type[EvolutionOperator] = GradientDescentEvolver,
    n_steps: int = 1000,
    dt: float = 1.0,
    evolver_kwargs: Optional[Dict[str, Any]] = None,
    record_trajectory: bool = True,
    verbose: bool = True
) -> SimulationResult:
    """
    Run simulation and return structured results with analysis.
    
    This is a convenience wrapper that returns a SimulationResult
    object with various analysis methods.
    
    Args:
        substrate: Initial substrate state
        constraints: List of constraints
        energy: Energy landscape
        evolver_cls: Evolution operator class
        n_steps: Number of steps
        dt: Time step size
        evolver_kwargs: Additional evolver arguments
        record_trajectory: Whether to record trajectory
        verbose: Whether to show progress
        
    Returns:
        SimulationResult object with analysis methods
    """
    # Initialize evolver
    evolver_kwargs = evolver_kwargs or {}
    evolver = evolver_cls(
        substrate=substrate,
        constraints=constraints,
        energy=energy,
        **evolver_kwargs
    )
    
    # Setup trajectory recording
    trajectory = []
    if record_trajectory:
        trajectory.append(substrate.clone())
    
    # Record initial state
    evolver.record_step()
    
    # Setup progress bar
    iterator = range(n_steps)
    if verbose:
        iterator = tqdm(iterator, desc="COEC Analysis")
    
    # Evolution loop
    for step in iterator:
        evolver.step(dt)
        evolver.record_step()
        
        if record_trajectory:
            trajectory.append(evolver.S.clone())
        
        if verbose:
            iterator.set_postfix({
                'E': f'{evolver.history["energy"][-1]:.3f}',
                'C': f'{evolver.history["constraint_satisfaction"][-1]:.3f}'
            })
    
    # Return structured results
    return SimulationResult(
        final_substrate=evolver.S,
        evolver=evolver,
        trajectory=trajectory if record_trajectory else None
    )