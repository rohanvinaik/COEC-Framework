"""
Entropy energy placeholder - To be implemented
"""
from ..core.energy import EnergyLandscape


class EntropyEnergy(EnergyLandscape):
    """
    Placeholder for entropy-based energy landscapes.
    
    Future implementation will include:
    - Shannon entropy
    - Boltzmann entropy
    - Information-theoretic measures
    - Cross-entropy with target distributions
    """
    
    def __init__(self, name: str = "EntropyEnergy"):
        super().__init__(name)
    
    def energy(self, s) -> float:
        """Placeholder implementation."""
        return 0.0  # Zero energy for now