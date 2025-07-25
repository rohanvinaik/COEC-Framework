"""
Protein Folding Example - SS-COEC (Static-Structural)

This example demonstrates how a protein-like chain finds its stable
folded configuration through constraint satisfaction.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from coec import Substrate, COECSystem
from coec.constraints import EnergeticConstraint, TopologicalConstraint, BoundaryConstraint
from coec.evolution import GradientDescentEvolver, MetropolisHastingsEvolver


def create_protein_system(n_residues: int = 20, method: str = "gradient"):
    """
    Create a simplified protein folding system.
    
    Args:
        n_residues: Number of amino acid residues
        method: Evolution method ("gradient" or "monte_carlo")
    """
    
    # Create substrate - a chain of residues in 3D space
    # Initialize in extended conformation
    initial_state = np.zeros((n_residues, 3))
    for i in range(n_residues):
        initial_state[i, 0] = i * 1.0  # Extended along x-axis
    
    substrate = Substrate(dimensions=3, size=n_residues, initial_state=initial_state)
    
    # Define constraints
    
    # 1. Energetic constraint - Lennard-Jones potential for hydrophobic interactions
    energy_constraint = EnergeticConstraint(
        name="hydrophobic",
        precision=1.0,
        potential="lennard_jones",
        parameters={
            "epsilon": 1.0,
            "sigma": 2.0,
            "max_energy": 50.0
        }
    )
    
    # 2. Topological constraint - maintain chain connectivity
    topology_constraint = TopologicalConstraint(
        name="backbone",
        precision=2.0,  # High precision - must maintain connectivity
        connectivity="chain",
        parameters={
            "bond_length": 1.0,
            "tolerance": 0.3
        }
    )
    
    # 3. Boundary constraint - keep protein in reasonable volume
    boundary_constraint = BoundaryConstraint(
        name="confinement",
        precision=0.5,
        boundary_type="sphere",
        parameters={
            "center": np.zeros(3),
            "radius": n_residues / 2
        }
    )
    
    constraints = [energy_constraint, topology_constraint, boundary_constraint]
    
    # Choose evolution operator
    if method == "gradient":
        evolver = GradientDescentEvolver(learning_rate=0.01, momentum=0.9)
    else:
        evolver = MetropolisHastingsEvolver(temperature=1.0, step_size=0.1)
    
    # Create COEC system
    system = COECSystem(
        substrate=substrate,
        constraints=constraints,
        evolver=evolver
    )
    
    return system


def visualize_folding(result, save_path: str = None):
    """
    Visualize the protein folding trajectory.
    """
    trajectory = result.trajectory
    n_steps, n_residues, _ = trajectory.shape
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # 3D structure plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('Final Folded Structure')
    
    # Plot final structure
    final_state = trajectory[-1]
    ax1.plot(final_state[:, 0], final_state[:, 1], final_state[:, 2], 
             'bo-', linewidth=2, markersize=6)
    
    # Highlight hydrophobic residues (mock - every 3rd residue)
    hydrophobic_indices = list(range(2, n_residues, 3))
    ax1.scatter(final_state[hydrophobic_indices, 0],
                final_state[hydrophobic_indices, 1],
                final_state[hydrophobic_indices, 2],
                c='red', s=100, label='Hydrophobic')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # Energy plot
    ax2 = fig.add_subplot(132)
    ax2.set_title('Energy Landscape')
    if 'energy_history' in result.metadata:
        ax2.plot(result.metadata['energy_history'])
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Energy')
        ax2.grid(True, alpha=0.3)
    
    # Constraint satisfaction plot
    ax3 = fig.add_subplot(133)
    ax3.set_title('Constraint Satisfaction')
    if 'constraint_history' in result.metadata:
        for name, history in result.metadata['constraint_history'].items():
            ax3.plot(history, label=name)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Satisfaction')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    
    return fig


def create_folding_animation(result, output_file: str = "protein_folding.gif"):
    """
    Create an animation of the folding process.
    """
    trajectory = result.trajectory
    n_steps = len(trajectory)
    
    # Subsample for faster animation
    subsample = max(1, n_steps // 100)
    frames = trajectory[::subsample]
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set consistent axes limits
    all_coords = trajectory.reshape(-1, 3)
    margin = 2
    ax.set_xlim([all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin])
    ax.set_ylim([all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin])
    ax.set_zlim([all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin])
    
    line, = ax.plot([], [], [], 'bo-', linewidth=2, markersize=6)
    
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line,
    
    def animate(i):
        state = frames[i]
        line.set_data(state[:, 0], state[:, 1])
        line.set_3d_properties(state[:, 2])
        ax.set_title(f'Step {i * subsample}')
        return line,
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(frames), interval=50, blit=True)
    
    anim.save(output_file, writer='pillow')
    plt.close()
    
    print(f"Animation saved to {output_file}")


def main():
    """
    Run the protein folding example.
    """
    print("Creating protein folding system...")
    system = create_protein_system(n_residues=30, method="gradient")
    
    print("Running folding simulation...")
    result = system.evolve(steps=2000)
    
    print(f"Final energy: {result.final_energy:.3f}")
    print("Final constraint satisfaction:")
    for name, satisfaction in result.constraint_satisfaction.items():
        print(f"  {name}: {satisfaction:.3f}")
    
    # Visualize results
    print("Generating visualization...")
    visualize_folding(result, save_path="protein_folding_result.png")
    
    # Create animation
    print("Creating animation...")
    create_folding_animation(result)
    
    print("Done!")


if __name__ == "__main__":
    main()
