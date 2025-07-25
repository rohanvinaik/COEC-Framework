"""
COEC Framework - Constraint-Oriented Emergent Computation

A substrate-independent framework for understanding computation as the trajectory 
of physical or biological systems through constrained state spaces.
"""

from .core.system import (
    Substrate,
    Constraint,
    EnergyLandscape,
    EvolutionOperator,
    InformationStructure,
    COECSystem,
    COECResult
)

__version__ = "0.1.0"
__author__ = "Rohan Vinaik"

__all__ = [
    "Substrate",
    "Constraint", 
    "EnergyLandscape",
    "EvolutionOperator",
    "InformationStructure",
    "COECSystem",
    "COECResult"
]
