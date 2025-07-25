"""
Basic tests for COEC core functionality.
"""

import numpy as np
import pytest
from coec import Substrate, COECSystem
from coec.constraints import EnergeticConstraint, TopologicalConstraint
from coec.evolution import GradientDescentEvolver


def test_substrate_creation():
    """Test substrate initialization."""
    substrate = Substrate(dimensions=3, size=10)
    assert substrate.state.shape == (10, 3)
    assert substrate.configuration_space_dims == 30


def test_constraint_satisfaction():
    """Test basic constraint satisfaction computation."""
    # Create a simple 2-particle system
    state = np.array([[0, 0, 0], [1, 0, 0]])
    substrate = Substrate(dimensions=3, size=2, initial_state=state)
    
    # Test energetic constraint
    energy_constraint = EnergeticConstraint(
        potential="harmonic",
        parameters={"k": 1.0, "r0": 1.0}
    )
    
    satisfaction = energy_constraint.satisfaction(substrate.get_state())
    assert 0 <= satisfaction <= 1


def test_system_evolution():
    """Test that system can evolve without errors."""
    # Create simple system
    substrate = Substrate(dimensions=2, size=5)
    constraints = [
        EnergeticConstraint(potential="harmonic"),
        TopologicalConstraint(connectivity="chain")
    ]
    evolver = GradientDescentEvolver(learning_rate=0.01)
    
    system = COECSystem(
        substrate=substrate,
        constraints=constraints,
        evolver=evolver
    )
    
    # Run short evolution
    result = system.evolve(steps=10)
    
    assert result.trajectory.shape[0] == 10
    assert result.final_state.shape == (5, 2)
    assert isinstance(result.final_energy, float)


if __name__ == "__main__":
    test_substrate_creation()
    test_constraint_satisfaction()
    test_system_evolution()
    print("All tests passed!")
