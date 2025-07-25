"""
Protein Folding Example - SS-COEC (Static-Structural)

This example demonstrates how COEC can model protein folding as
constraint-driven computation where the output is a stable 3D structure.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from coec import (
    EuclideanSubstrate,
    Constraint,
    EnergyLandscape,
    GradientDescentEvolver,
    run_simulation_with_analysis
)


class ProteinSubstrate(EuclideanSubstrate):
    """
    Specialized substrate for protein representation.
    State vector contains 3D coordinates for each amino acid.
    """
    
    def __init__(self, n_residues: int, initial_coords: np.ndarray = None):
        """
        Initialize protein substrate.
        
        Args:
            n_residues: Number of amino acids
            initial_coords: Initial 3D coordinates (n_residues, 3)
        """
        self.n_residues = n_residues
        
        if initial_coords is None:
            # Random coil initialization
            initial_coords = np.random.randn(n_residues, 3) * 5
        
        # Flatten to 1D for EuclideanSubstrate
        super().__init__(state=initial_coords.flatten())
    
    def get_coords(self) -> np.ndarray:
        """Get 3D coordinates of all residues."""
        return self.state.reshape(self.n_residues, 3)
    
    def set_coords(self, coords: np.ndarray) -> None:
        """Set 3D coordinates of all residues."""
        self.state = coords.flatten()


class BondLengthConstraint(Constraint):
    """
    Constraint maintaining proper bond lengths between consecutive residues.
    """
    
    def __init__(self, target_length: float = 3.8, tolerance: float = 0.2):
        """
        Initialize bond length constraint.
        
        Args:
            target_length: Target C-alpha to C-alpha distance (Angstroms)
            tolerance: Acceptable deviation from target
        """
        super().__init__(precision=10.0, name="BondLength")
        self.target_length = target_length
        self.tolerance = tolerance
    
    def evaluate(self, s: ProteinSubstrate) -> float:
        """
        Evaluate bond length satisfaction.
        Returns 1.0 if all bonds are within tolerance.
        """
        coords = s.get_coords()
        satisfaction = 0.0
        
        for i in range(len(coords) - 1):
            dist = np.linalg.norm(coords[i+1] - coords[i])
            deviation = abs(dist - self.target_length)
            
            # Sigmoid-like satisfaction
            bond_sat = 1.0 / (1.0 + (deviation / self.tolerance)**2)
            satisfaction += bond_sat
        
        return satisfaction / (len(coords) - 1)
    
    def grad(self, s: ProteinSubstrate) -> np.ndarray:
        """Compute gradient for bond length constraint."""
        coords = s.get_coords()
        grad = np.zeros_like(coords)
        
        for i in range(len(coords) - 1):
            vec = coords[i+1] - coords[i]
            dist = np.linalg.norm(vec)
            
            if dist > 1e-6:
                # Force to restore proper bond length
                deviation = dist - self.target_length
                force = -2 * deviation / (self.tolerance**2 + deviation**2)**2
                direction = vec / dist
                
                grad[i] -= force * direction
                grad[i+1] += force * direction
        
        return grad.flatten()


class HydrophobicConstraint(Constraint):
    """
    Constraint encouraging hydrophobic residues to cluster.
    """
    
    def __init__(self, hydrophobic_indices: List[int], radius: float = 8.0):
        """
        Initialize hydrophobic constraint.
        
        Args:
            hydrophobic_indices: Indices of hydrophobic residues
            radius: Interaction radius for hydrophobic effect
        """
        super().__init__(precision=5.0, name="Hydrophobic")
        self.hydrophobic_indices = hydrophobic_indices
        self.radius = radius
    
    def evaluate(self, s: ProteinSubstrate) -> float:
        """
        Evaluate hydrophobic clustering.
        Higher satisfaction when hydrophobic residues are close.
        """
        coords = s.get_coords()
        hydro_coords = coords[self.hydrophobic_indices]
        
        if len(hydro_coords) < 2:
            return 1.0
        
        # Compute pairwise interactions
        satisfaction = 0.0
        n_pairs = 0
        
        for i in range(len(hydro_coords)):
            for j in range(i + 1, len(hydro_coords)):
                dist = np.linalg.norm(hydro_coords[i] - hydro_coords[j])
                # Sigmoid interaction
                interaction = 1.0 / (1.0 + (dist / self.radius)**2)
                satisfaction += interaction
                n_pairs += 1
        
        return satisfaction / n_pairs if n_pairs > 0 else 0.0


class LennardJonesEnergy(EnergyLandscape):
    """
    Lennard-Jones potential for non-bonded interactions.
    """
    
    def __init__(self, epsilon: float = 1.0, sigma: float = 3.4):
        """
        Initialize LJ potential.
        
        Args:
            epsilon: Depth of potential well
            sigma: Distance at which potential is zero
        """
        super().__init__(name="LennardJones")
        self.epsilon = epsilon
        self.sigma = sigma
    
    def energy(self, s: ProteinSubstrate) -> float:
        """
        Compute total Lennard-Jones energy.
        """
        coords = s.get_coords()
        total_energy = 0.0
        
        # Sum over all non-bonded pairs (skip adjacent residues)
        for i in range(len(coords)):
            for j in range(i + 2, len(coords)):  # Skip i+1 (bonded)
                r = np.linalg.norm(coords[i] - coords[j])
                
                if r > 0.1:  # Avoid singularity
                    # LJ potential: 4ε[(σ/r)^12 - (σ/r)^6]
                    sr6 = (self.sigma / r) ** 6
                    total_energy += 4 * self.epsilon * (sr6**2 - sr6)
        
        return total_energy
    
    def grad(self, s: ProteinSubstrate) -> np.ndarray:
        """Compute gradient of LJ energy."""
        coords = s.get_coords()
        grad = np.zeros_like(coords)
        
        for i in range(len(coords)):
            for j in range(i + 2, len(coords)):
                vec = coords[j] - coords[i]
                r = np.linalg.norm(vec)
                
                if r > 0.1:
                    # Force from LJ: F = -dV/dr * r_hat
                    sr6 = (self.sigma / r) ** 6
                    force_mag = 24 * self.epsilon / r * (2 * sr6**2 - sr6)
                    force = force_mag * vec / r
                    
                    grad[i] += force
                    grad[j] -= force
        
        return grad.flatten()


def visualize_protein(substrate: ProteinSubstrate, title: str = "Protein Structure"):
    """Visualize protein structure in 3D."""
    coords = substrate.get_coords()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot backbone
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 
            'b-', linewidth=2, label='Backbone')
    
    # Plot residues
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
              c=np.arange(len(coords)), cmap='viridis', s=100)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    return fig


def main():
    """Run protein folding simulation."""
    print("=== COEC Protein Folding Example (SS-COEC) ===\n")
    
    # Setup protein
    n_residues = 20
    hydrophobic_positions = [2, 5, 8, 11, 14, 17]  # Every 3rd residue
    
    # Initialize substrate
    protein = ProteinSubstrate(n_residues)
    print(f"Initialized {n_residues}-residue protein")
    
    # Define constraints
    constraints = [
        BondLengthConstraint(target_length=3.8, tolerance=0.2),
        HydrophobicConstraint(hydrophobic_positions, radius=8.0)
    ]
    
    # Define energy
    energy = LennardJonesEnergy(epsilon=0.5, sigma=3.4)
    
    # Run simulation
    print("\nRunning COEC simulation...")
    result = run_simulation_with_analysis(
        substrate=protein,
        constraints=constraints,
        energy=energy,
        evolver_cls=GradientDescentEvolver,
        n_steps=1000,
        evolver_kwargs={'lr_E': 0.01, 'lr_C': 0.02, 'momentum': 0.9},
        verbose=True
    )
    
    # Analyze results
    print("\n=== Results ===")
    print(f"Final energy: {result.final_energy:.3f}")
    print(f"Final constraint satisfaction: {result.final_satisfaction:.3f}")
    
    satisfactions = result.get_constraint_satisfactions()
    for name, value in satisfactions.items():
        print(f"  {name}: {value:.3f}")
    
    # Visualize
    if result.trajectory:
        # Show initial and final structures
        fig1 = visualize_protein(result.trajectory[0], "Initial Structure")
        fig2 = visualize_protein(result.final_substrate, "Final Structure")
        
        # Plot convergence
        history = result.get_history()
        fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(history['energy'])
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy Minimization')
        ax1.grid(True)
        
        ax2.plot(history['constraint_satisfaction'])
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Total Constraint Satisfaction')
        ax2.set_title('Constraint Satisfaction')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    print("\nProtein folding simulation complete!")
    print("This demonstrates SS-COEC: computation producing stable structure")


if __name__ == "__main__":
    main()