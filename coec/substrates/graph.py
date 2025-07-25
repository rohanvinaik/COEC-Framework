"""
Graph substrate - Network-based state representation
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Optional, Union, Dict, Any

from ..core.substrate import Substrate


class GraphSubstrate(Substrate['GraphSubstrate']):
    """
    Graph-based substrate for network computations.
    
    This substrate represents states as graphs where computation
    emerges from network topology and node/edge attributes.
    Useful for:
    - Social networks
    - Biological networks (protein interactions, metabolic)
    - Infrastructure networks
    - Neural architectures
    
    References:
        - §3.7: GCT-COEC (Graph-Constrained Topology)
        - §9.3: Network-based applications
    """
    
    def __init__(
        self,
        graph: Optional[nx.Graph] = None,
        node_dim: int = 1,
        edge_dim: int = 0,
        directed: bool = False
    ):
        """
        Initialize graph substrate.
        
        Args:
            graph: Initial NetworkX graph (if None, creates empty graph)
            node_dim: Dimension of node attribute vectors
            edge_dim: Dimension of edge attribute vectors (0 = no edge attrs)
            directed: Whether to use directed graph
        """
        if graph is None:
            graph = nx.DiGraph() if directed else nx.Graph()
        
        self.graph = graph
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.directed = directed
        
        # Initialize node attributes if not present
        self._initialize_attributes()
        
        # State is the graph itself
        super().__init__(state=graph)
    
    def _initialize_attributes(self) -> None:
        """Initialize node and edge attributes with random values."""
        # Initialize node attributes
        for node in self.graph.nodes():
            if 'state' not in self.graph.nodes[node]:
                self.graph.nodes[node]['state'] = np.random.randn(self.node_dim)
        
        # Initialize edge attributes if needed
        if self.edge_dim > 0:
            for u, v in self.graph.edges():
                if 'weight' not in self.graph.edges[u, v]:
                    self.graph.edges[u, v]['weight'] = np.random.randn(self.edge_dim)
    
    def _clone_state(self) -> nx.Graph:
        """Create a deep copy of the graph."""
        return self.graph.copy()
    
    def distance(self, other: GraphSubstrate) -> float:
        """
        Compute distance between graph substrates.
        
        Uses a combination of:
        - Graph edit distance (structural)
        - Node attribute distance (L2)
        - Edge attribute distance (if applicable)
        
        Args:
            other: Another GraphSubstrate
            
        Returns:
            Combined distance metric
        """
        if not isinstance(other, GraphSubstrate):
            raise TypeError(f"Cannot compute distance to {type(other)}")
        
        # Structural distance (normalized edit distance)
        try:
            # For small graphs, exact edit distance
            if len(self.graph) < 10 and len(other.graph) < 10:
                edit_dist = nx.graph_edit_distance(self.graph, other.graph)
                max_size = max(len(self.graph), len(other.graph))
                structural_dist = edit_dist / (max_size + 1)
            else:
                # For larger graphs, use approximation
                structural_dist = self._approximate_structural_distance(other)
        except:
            structural_dist = 1.0  # Maximum distance if comparison fails
        
        # Node attribute distance
        node_dist = self._node_attribute_distance(other)
        
        # Combine distances (can be weighted differently)
        return float(0.5 * structural_dist + 0.5 * node_dist)
    
    def _approximate_structural_distance(self, other: GraphSubstrate) -> float:
        """
        Approximate structural distance using graph statistics.
        
        Args:
            other: Another GraphSubstrate
            
        Returns:
            Approximate distance in [0, 1]
        """
        # Compare basic graph properties
        props1 = self._compute_graph_properties()
        props2 = other._compute_graph_properties()
        
        # Normalized differences
        diffs = []
        for key in props1:
            if props1[key] + props2[key] > 0:
                diff = abs(props1[key] - props2[key]) / (props1[key] + props2[key])
                diffs.append(diff)
        
        return np.mean(diffs) if diffs else 1.0
    
    def _compute_graph_properties(self) -> Dict[str, float]:
        """Compute basic graph properties for comparison."""
        g = self.graph
        n = len(g)
        
        if n == 0:
            return {
                'nodes': 0,
                'edges': 0,
                'density': 0,
                'avg_degree': 0,
                'clustering': 0
            }
        
        return {
            'nodes': n,
            'edges': g.number_of_edges(),
            'density': nx.density(g),
            'avg_degree': sum(dict(g.degree()).values()) / n,
            'clustering': nx.average_clustering(g.to_undirected() if self.directed else g)
        }
    
    def _node_attribute_distance(self, other: GraphSubstrate) -> float:
        """
        Compute distance between node attributes.
        
        Args:
            other: Another GraphSubstrate
            
        Returns:
            Average L2 distance between matched nodes
        """
        # Simple approach: compare nodes with same labels
        common_nodes = set(self.graph.nodes()) & set(other.graph.nodes())
        
        if not common_nodes:
            return 1.0  # Maximum distance if no common nodes
        
        distances = []
        for node in common_nodes:
            state1 = self.graph.nodes[node].get('state', np.zeros(self.node_dim))
            state2 = other.graph.nodes[node].get('state', np.zeros(other.node_dim))
            
            if len(state1) == len(state2):
                distances.append(np.linalg.norm(state1 - state2))
        
        if distances:
            # Normalize by dimension
            return np.mean(distances) / np.sqrt(self.node_dim)
        else:
            return 1.0
    
    def dimension(self) -> int:
        """
        Return the effective dimensionality of the graph substrate.
        
        Returns:
            Total dimension (nodes * node_dim + edges * edge_dim)
        """
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        
        return n_nodes * self.node_dim + n_edges * self.edge_dim
    
    def add_node(self, node_id: Any, state: Optional[np.ndarray] = None) -> None:
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique identifier for the node
            state: Initial state vector (random if None)
        """
        if state is None:
            state = np.random.randn(self.node_dim)
        
        self.graph.add_node(node_id, state=state)
    
    def add_edge(
        self,
        u: Any,
        v: Any,
        weight: Optional[Union[float, np.ndarray]] = None
    ) -> None:
        """
        Add an edge to the graph.
        
        Args:
            u: Source node
            v: Target node
            weight: Edge weight (scalar or vector)
        """
        if self.edge_dim > 0 and weight is None:
            weight = np.random.randn(self.edge_dim)
        
        self.graph.add_edge(u, v, weight=weight)
    
    def get_node_states(self) -> np.ndarray:
        """
        Get all node states as a matrix.
        
        Returns:
            Matrix of shape (n_nodes, node_dim)
        """
        nodes = sorted(self.graph.nodes())
        states = []
        
        for node in nodes:
            state = self.graph.nodes[node].get('state', np.zeros(self.node_dim))
            states.append(state)
        
        return np.array(states)
    
    def set_node_states(self, states: np.ndarray) -> None:
        """
        Set all node states from a matrix.
        
        Args:
            states: Matrix of shape (n_nodes, node_dim)
        """
        nodes = sorted(self.graph.nodes())
        
        if len(states) != len(nodes):
            raise ValueError(
                f"State matrix has {len(states)} rows but graph has {len(nodes)} nodes"
            )
        
        for node, state in zip(nodes, states):
            self.graph.nodes[node]['state'] = state
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get adjacency matrix of the graph.
        
        Returns:
            Adjacency matrix
        """
        return nx.adjacency_matrix(self.graph).toarray()
    
    def get_laplacian_matrix(self) -> np.ndarray:
        """
        Get Laplacian matrix of the graph.
        
        Returns:
            Laplacian matrix
        """
        return nx.laplacian_matrix(self.graph).toarray()
    
    def compute_graph_metrics(self) -> Dict[str, Any]:
        """
        Compute various graph metrics.
        
        Returns:
            Dictionary of graph metrics
        """
        g = self.graph
        
        metrics = {
            'num_nodes': g.number_of_nodes(),
            'num_edges': g.number_of_edges(),
            'density': nx.density(g),
            'is_connected': nx.is_connected(g.to_undirected() if self.directed else g)
        }
        
        if g.number_of_nodes() > 0:
            metrics['avg_degree'] = sum(dict(g.degree()).values()) / g.number_of_nodes()
            
            if not self.directed:
                metrics['clustering_coefficient'] = nx.average_clustering(g)
                if metrics['is_connected']:
                    metrics['diameter'] = nx.diameter(g)
                    metrics['avg_path_length'] = nx.average_shortest_path_length(g)
        
        return metrics
    
    def __repr__(self) -> str:
        """String representation of the graph substrate."""
        return (
            f"GraphSubstrate("
            f"nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()}, "
            f"node_dim={self.node_dim}, "
            f"directed={self.directed})"
        )