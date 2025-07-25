"""
Setup configuration for COEC Framework
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="coec-framework",
    version="0.1.0",
    author="Rohan Vinaik",
    author_email="",
    description="Constraint-Oriented Emergent Computation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rohanvinaik/COEC-Framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "networkx>=2.6.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
            "pre-commit>=2.17.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.17.0",
        ],
        "quantum": [
            "qiskit>=0.36.0",
        ],
        "ml": [
            "torch>=1.11.0",
            "scikit-learn>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "coec=coec.cli:main",
        ],
    },
)