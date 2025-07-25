"""
COEC Framework - Constraint-Oriented Emergent Computation

A substrate-independent framework describing computation as the trajectory 
of physical or biological systems through constrained state spaces.
"""

__version__ = "0.1.0"

# Core abstractions
from .core.substrate import Substrate
from .core.constraint import Constraint
from .core.energy import EnergyLandscape
from .core.evolution import EvolutionOperator
from .core.residual import ResidualFunction

# Common implementations
from .substrates.euclidean import EuclideanSubstrate
from .substrates.graph import GraphSubstrate
from .constraints.linear import LinearConstraint
from .constraints.topological import TopologicalConstraint
from .energy.quadratic import QuadraticEnergy
from .energy.entropy import EntropyEnergy
from .evolvers.gradient import GradientDescentEvolver
from .evolvers.metropolis import MetropolisHastingsEvolver

# Simulation utilities
from .simulation import run_simulation

__all__ = [
    # Core
    "Substrate",
    "Constraint",
    "EnergyLandscape",
    "EvolutionOperator",
    "ResidualFunction",
    # Implementations
    "EuclideanSubstrate",
    "GraphSubstrate",
    "LinearConstraint",
    "TopologicalConstraint",
    "QuadraticEnergy",
    "EntropyEnergy",
    "GradientDescentEvolver",
    "MetropolisHastingsEvolver",
    # Utils
    "run_simulation",
]