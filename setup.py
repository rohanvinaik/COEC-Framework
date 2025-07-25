from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="coec-framework",
    version="0.1.0",
    author="Rohan Vinaik",
    author_email="your.email@example.com",
    description="Constraint-Oriented Emergent Computation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rohanvinaik/COEC-Framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "networkx>=2.6.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": ["pytest>=6.2.0", "black>=21.6b0", "flake8>=3.9.0"],
        "viz": ["plotly>=5.3.0", "streamlit>=0.86.0"],
        "jupyter": ["jupyter>=1.0.0", "ipywidgets>=7.6.0"],
    },
)
