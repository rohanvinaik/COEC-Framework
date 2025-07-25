"""
Visualization tools for COEC systems.

Provides interactive and static visualizations of constraint networks,
energy landscapes, and system evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from typing import List, Dict, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class COECVisualizer:
    """
    Main visualization class for COEC systems.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize
        self.color_palette = {
            'energetic': '#FF6B6B',
            'topological': '#4ECDC4',
            'informational': '#45B7D1',
            'boundary': '#96CEB4',
            'default': '#95A5A6'
        }
    
    def plot_constraint_network(self, constraints: List, 
                               precision_weights: Optional[Dict[str, float]] = None,
                               layout: str = 'spring'):
        """
        Visualize the constraint network as a graph.
        
        Node size represents precision weight, edges show interactions.
        """
        G = nx.Graph()
        
        # Add nodes for each constraint
        for constraint in constraints:
            G.add_node(constraint.name, 
                      type=constraint.__class__.__name__,
                      precision=constraint.precision)
        
        # Add edges based on constraint interactions
        # For now, we'll create a fully connected graph
        # In practice, this would be based on actual interactions
        for i, c1 in enumerate(constraints):
            for j, c2 in enumerate(constraints[i+1:], i+1):
                G.add_edge(c1.name, c2.name)
        
        # Create layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G)
        
        # Plot
        plt.figure(figsize=self.figsize)
        
        # Draw nodes
        for node in G.nodes():
            node_data = G.nodes[node]
            color = self.color_palette.get(
                node_data['type'].replace('Constraint', '').lower(), 
                self.color_palette['default']
            )
            size = 1000 * node_data['precision']
            
            nx.draw_networkx_nodes(G, pos, [node], 
                                 node_color=color,
                                 node_size=size,
                                 alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        plt.title("Constraint Network Topology", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_energy_landscape_2d(self, energy_landscape, 
                                bounds: List[Tuple[float, float]] = [(-5, 5), (-5, 5)],
                                resolution: int = 50):
        """
        Visualize 2D energy landscape as a contour plot.
        """
        x = np.linspace(bounds[0][0], bounds[0][1], resolution)
        y = np.linspace(bounds[1][0], bounds[1][1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Compute energy for each point
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                state = np.array([[X[i, j], Y[i, j]]])
                Z[i, j] = energy_landscape.compute_energy(state)
        
        # Create figure
        fig = go.Figure()
        
        # Add contour plot
        fig.add_trace(go.Contour(
            x=x, y=y, z=Z,
            colorscale='Viridis',
            contours=dict(
                showlabels=True,
                labelfont=dict(size=12, color='white')
            )
        ))
        
        fig.update_layout(
            title="Energy Landscape",
            xaxis_title="X",
            yaxis_title="Y",
            width=800,
            height=600
        )
        
        return fig
    
    def plot_trajectory_2d(self, trajectory: np.ndarray, 
                          energy_landscape=None,
                          show_energy_contours: bool = True):
        """
        Plot system trajectory in 2D state space.
        """
        if trajectory.shape[2] != 2:
            raise ValueError("This method only works for 2D trajectories")
        
        fig = plt.figure(figsize=self.figsize)
        ax = plt.gca()
        
        # Plot energy contours if provided
        if energy_landscape and show_energy_contours:
            bounds = [
                (trajectory[:, :, 0].min() - 1, trajectory[:, :, 0].max() + 1),
                (trajectory[:, :, 1].min() - 1, trajectory[:, :, 1].max() + 1)
            ]
            
            x = np.linspace(bounds[0][0], bounds[0][1], 50)
            y = np.linspace(bounds[1][0], bounds[1][1], 50)
            X, Y = np.meshgrid(x, y)
            
            Z = np.zeros_like(X)
            for i in range(50):
                for j in range(50):
                    state = np.array([[X[i, j], Y[i, j]]])
                    Z[i, j] = energy_landscape.compute_energy(state)
            
            contour = ax.contour(X, Y, Z, levels=20, alpha=0.3, cmap='viridis')
            ax.clabel(contour, inline=True, fontsize=8)
        
        # Plot trajectories for each particle
        n_particles = trajectory.shape[1]
        colors = plt.cm.rainbow(np.linspace(0, 1, n_particles))
        
        for i in range(n_particles):
            # Plot trajectory
            ax.plot(trajectory[:, i, 0], trajectory[:, i, 1], 
                   color=colors[i], alpha=0.5, linewidth=1)
            
            # Mark start and end
            ax.scatter(trajectory[0, i, 0], trajectory[0, i, 1], 
                      color=colors[i], marker='o', s=100, 
                      edgecolor='black', label=f'Particle {i} start')
            ax.scatter(trajectory[-1, i, 0], trajectory[-1, i, 1], 
                      color=colors[i], marker='s', s=100, 
                      edgecolor='black', label=f'Particle {i} end')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('System Trajectory in State Space')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_evolution_dashboard(self, result, constraints: List):
        """
        Create a comprehensive dashboard showing system evolution.
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('System Configuration', 'Energy Evolution',
                          'Constraint Satisfaction', 'Phase Space'),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        trajectory = result.trajectory
        metadata = result.metadata
        
        # 1. Final configuration (3D if possible, 2D projection otherwise)
        final_state = result.final_state
        if final_state.shape[1] >= 3:
            fig.add_trace(
                go.Scatter3d(
                    x=final_state[:, 0],
                    y=final_state[:, 1],
                    z=final_state[:, 2],
                    mode='markers+lines',
                    marker=dict(size=8, color='red'),
                    line=dict(width=3),
                    name='Final Structure'
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=final_state[:, 0],
                    y=final_state[:, 1] if final_state.shape[1] > 1 else np.zeros_like(final_state[:, 0]),
                    mode='markers+lines',
                    marker=dict(size=10, color='red'),
                    line=dict(width=2),
                    name='Final Structure'
                ),
                row=1, col=1
            )
        
        # 2. Energy evolution
        if 'energy_history' in metadata:
            fig.add_trace(
                go.Scatter(
                    y=metadata['energy_history'],
                    mode='lines',
                    name='Energy',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=2
            )
        
        # 3. Constraint satisfaction over time
        if 'constraint_history' in metadata:
            for name, history in metadata['constraint_history'].items():
                color = self.color_palette.get(
                    name.replace('Constraint', '').lower(),
                    self.color_palette['default']
                )
                fig.add_trace(
                    go.Scatter(
                        y=history,
                        mode='lines',
                        name=name,
                        line=dict(width=2)
                    ),
                    row=2, col=1
                )
        
        # 4. Phase space (if 2D)
        if trajectory.shape[2] >= 2:
            # Plot trajectory of first particle
            fig.add_trace(
                go.Scatter(
                    x=trajectory[:, 0, 0],
                    y=trajectory[:, 0, 1],
                    mode='lines',
                    name='Phase Trajectory',
                    line=dict(color='green', width=1),
                    opacity=0.7
                ),
                row=2, col=2
            )
            
            # Mark start and end
            fig.add_trace(
                go.Scatter(
                    x=[trajectory[0, 0, 0]],
                    y=[trajectory[0, 0, 1]],
                    mode='markers',
                    marker=dict(size=10, color='green', symbol='circle'),
                    name='Start',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[trajectory[-1, 0, 0]],
                    y=[trajectory[-1, 0, 1]],
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='square'),
                    name='End',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="COEC System Evolution Dashboard"
        )
        
        # Update axes
        fig.update_xaxes(title_text="Step", row=1, col=2)
        fig.update_yaxes(title_text="Energy", row=1, col=2)
        fig.update_xaxes(title_text="Step", row=2, col=1)
        fig.update_yaxes(title_text="Satisfaction", row=2, col=1)
        fig.update_xaxes(title_text="X", row=2, col=2)
        fig.update_yaxes(title_text="Y", row=2, col=2)
        
        return fig
    
    def create_evolution_animation(self, trajectory: np.ndarray, 
                                 output_file: str = "evolution.gif",
                                 interval: int = 50):
        """
        Create an animated visualization of system evolution.
        """
        fig = plt.figure(figsize=(8, 8))
        
        if trajectory.shape[2] == 2:
            ax = plt.gca()
            self._create_2d_animation(ax, trajectory, interval, output_file)
        elif trajectory.shape[2] == 3:
            ax = fig.add_subplot(111, projection='3d')
            self._create_3d_animation(ax, trajectory, interval, output_file)
        else:
            raise ValueError("Can only animate 2D or 3D trajectories")
        
        return fig
    
    def _create_2d_animation(self, ax, trajectory, interval, output_file):
        """Helper for 2D animations."""
        n_steps, n_particles, _ = trajectory.shape
        
        # Set up the plot
        all_x = trajectory[:, :, 0].flatten()
        all_y = trajectory[:, :, 1].flatten()
        margin = 0.1 * max(all_x.max() - all_x.min(), all_y.max() - all_y.min())
        
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Initialize lines and points
        lines = []
        points = []
        colors = plt.cm.rainbow(np.linspace(0, 1, n_particles))
        
        for i in range(n_particles):
            line, = ax.plot([], [], color=colors[i], alpha=0.5)
            point, = ax.plot([], [], 'o', color=colors[i], markersize=8)
            lines.append(line)
            points.append(point)
        
        def init():
            for line, point in zip(lines, points):
                line.set_data([], [])
                point.set_data([], [])
            return lines + points
        
        def animate(frame):
            for i, (line, point) in enumerate(zip(lines, points)):
                # Update trajectory
                line.set_data(trajectory[:frame+1, i, 0], 
                            trajectory[:frame+1, i, 1])
                # Update current position
                point.set_data([trajectory[frame, i, 0]], 
                             [trajectory[frame, i, 1]])
            
            ax.set_title(f'Step {frame}/{n_steps-1}')
            return lines + points
        
        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=n_steps, interval=interval, blit=True)
        
        anim.save(output_file, writer='pillow')
        print(f"Animation saved to {output_file}")
    
    def _create_3d_animation(self, ax, trajectory, interval, output_file):
        """Helper for 3D animations."""
        # Similar to 2D but with 3D plotting
        # Implementation left as exercise
        pass


def demonstrate_visualization():
    """
    Demonstrate visualization capabilities.
    """
    from coec import Substrate, COECSystem
    from coec.constraints import EnergeticConstraint, TopologicalConstraint
    from coec.evolution import GradientDescentEvolver
    
    print("Creating visualization demo...")
    
    # Create a simple system
    substrate = Substrate(dimensions=2, size=5)
    constraints = [
        EnergeticConstraint(name="attraction", potential="harmonic"),
        TopologicalConstraint(name="chain", connectivity="chain")
    ]
    evolver = GradientDescentEvolver(learning_rate=0.05)
    
    system = COECSystem(substrate, constraints, evolver=evolver)
    result = system.evolve(steps=100)
    
    # Create visualizer
    viz = COECVisualizer()
    
    # 1. Constraint network
    print("1. Plotting constraint network...")
    fig1 = viz.plot_constraint_network(constraints)
    plt.savefig("constraint_network.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Trajectory plot
    print("2. Plotting trajectory...")
    fig2 = viz.plot_trajectory_2d(result.trajectory)
    plt.savefig("trajectory.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Evolution dashboard (using plotly)
    print("3. Creating evolution dashboard...")
    fig3 = viz.plot_evolution_dashboard(result, constraints)
    fig3.write_html("evolution_dashboard.html")
    
    print("Visualizations saved!")


if __name__ == "__main__":
    demonstrate_visualization()
