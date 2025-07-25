# COEC Framework

Constraint-Oriented Emergent Computation (COEC) - A Formal Framework Bridging Biology and Computation

## Overview

COEC is a substrate-independent framework describing computation as the trajectory of physical or biological systems through constrained state spaces. In COEC, computation emerges not through discrete logic or symbolic manipulation, but through systems undergoing entropy-driven transitions within boundary conditions.

## 🚀 New: Advanced Techniques Integration

The framework now includes cutting-edge computational techniques that provide 10-100x performance improvements:

- **Kolmogorov-Arnold Networks (KAN)**: Interpretable spline-based learning with 50-100x compression
- **Entropy-Based Compute Governance**: Skip deterministic computations, allocate resources dynamically
- **Enhanced Hyperdimensional Computing**: Orthogonal error separation, Hamming LUT optimization
- **Physics-Guided Constraints**: Deterministic physics handles 90-97% of computation
- **Distributed & Federated COEC**: Privacy-preserving collaborative computation
- **Topological Data Analysis**: Ensure physical validity, guide compression

See the [Advanced Techniques Guide](docs/advanced_techniques/README.md) for details.

## Key Concepts

### 7-Tuple Ontology (§2.1)

The formal COEC system is defined as:

**COEC** = (Ω_S, C, E, Φ, Ω_{S|C}, R, T)

Where:
- **Ω_S**: Substrate space (configuration space of states)
- **C**: Set of constraints {c₁, c₂, ..., cₙ}
- **E**: Energy landscape
- **Φ**: Evolution operator
- **Ω_{S|C}**: Constrained state space
- **R**: Residual function (computational output)
- **T**: Timescale specification

### COEC Classes

1. **SS-COEC** (Static-Structural): Systems producing stable structural outputs
2. **DB-COEC** (Dynamic-Behavioral): Systems producing stable temporal patterns
3. **DM-COEC** (Distributed-Multiplicative): Computation from multiple interacting subsystems
4. **AP-COEC** (Adaptive-Plastic): Systems that modify their own constraints
5. **PP-COEC** (Predictive-Probabilistic): Systems using internal models
6. **GCT-COEC** (Graph-Constrained Topology): Systems using graph metrics
7. **TDA-COEC** (Topological Data Analysis): Systems using topological features
8. **Cat-COEC** (Catalytic): Systems using transient external memory

## Installation

```bash
# Clone the repository
git clone https://github.com/rohanvinaik/COEC-Framework.git
cd COEC-Framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```python
from coec import EuclideanSubstrate, LinearConstraint, QuadraticEnergy
from coec.evolvers import GradientDescentEvolver
from coec.simulation import run_simulation

# Create substrate
substrate = EuclideanSubstrate(state=np.random.randn(4))

# Define energy landscape
energy = QuadraticEnergy(Q=np.eye(4))

# Define constraints
constraints = [
    LinearConstraint(w=np.array([1, 0, 0, 0]), b=0.5, precision=2.0),
    LinearConstraint(w=np.array([0, 1, 0, 0]), b=-0.2, precision=1.0),
]

# Run simulation
final_state = run_simulation(
    substrate=substrate,
    constraints=constraints,
    energy=energy,
    evolver_cls=GradientDescentEvolver,
    n_steps=1000
)
```

## Project Structure

```
coec-framework/
├── coec/                      # Core package
│   ├── __init__.py
│   ├── core/                  # Core abstractions
│   │   ├── substrate.py       # Substrate base class
│   │   ├── constraint.py      # Constraint base class
│   │   ├── energy.py          # Energy landscape base
│   │   └── evolution.py       # Evolution operator base
│   ├── advanced/              # Advanced techniques (NEW)
│   │   ├── kan_networks.py    # Kolmogorov-Arnold Networks
│   │   ├── entropy_governance.py # Entropy-based compute
│   │   ├── hdcomputing_enhanced.py # Enhanced HD computing
│   │   ├── physics_constraints.py # Physics-guided constraints
│   │   ├── distributed_coec.py # Federated & distributed
│   │   └── topological_analysis.py # TDA enhancements
│   ├── substrates/            # Substrate implementations
│   │   ├── euclidean.py       # Euclidean space
│   │   ├── graph.py           # Graph-based substrates
│   │   └── quantum.py         # Quantum substrates
│   ├── constraints/           # Constraint types
│   │   ├── linear.py          # Linear constraints
│   │   ├── topological.py     # Topological constraints
│   │   └── adaptive.py        # Adaptive constraints
│   ├── energy/                # Energy landscapes
│   │   ├── quadratic.py       # Quadratic energy
│   │   ├── entropy.py         # Information-theoretic
│   │   └── composite.py       # Composite landscapes
│   ├── evolvers/              # Evolution operators
│   │   ├── gradient.py        # Gradient descent
│   │   ├── metropolis.py      # Metropolis-Hastings
│   │   └── quantum.py         # Quantum evolution
│   ├── residuals/             # Residual functions
│   │   ├── structural.py      # Structural outputs
│   │   └── behavioral.py      # Behavioral patterns
│   └── utils/                 # Utilities
│       ├── visualization.py   # Plotting tools
│       └── metrics.py         # Analysis metrics
├── examples/                  # Example implementations
│   ├── protein_folding.py     # SS-COEC example
│   ├── circadian_rhythm.py    # DB-COEC example
│   ├── swarm_behavior.py      # DM-COEC example
│   └── neural_plasticity.py   # AP-COEC example
├── tests/                     # Test suite
├── docs/                      # Documentation
├── requirements.txt
├── setup.py
└── README.md
```

## Documentation

- [Mathematical Foundations](docs/mathematical_foundations.md)
- [API Reference](docs/api_reference.md)
- [Implementation Guide](docs/implementation_guide.md)
- [Case Studies](docs/case_studies.md)
- [Advanced Techniques Guide](docs/advanced_techniques/README.md) **(NEW)**

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## References

- Constraint-Oriented Emergent Computation: A Formal Framework Bridging Biology and Computation
- [Project Knowledge Base](docs/references/)

## Contact

- Repository: https://github.com/rohanvinaik/COEC-Framework
- Issues: https://github.com/rohanvinaik/COEC-Framework/issues