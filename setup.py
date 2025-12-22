"""
Setup script for Müntz-Szász Networks.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="muntz-szasz-networks",
    version="0.1.0",
    author="Gnankan Landry Regis N'guessan",
    author_email="rnguessan@aimsric.org",
    description="Neural network architecture with learnable fractional power bases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[username]/muntz-szasz-networks",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "tqdm>=4.60.0",
        ],
    },
    keywords=[
        "neural networks",
        "deep learning",
        "approximation theory",
        "physics-informed neural networks",
        "PINNs",
        "singular functions",
        "power functions",
        "Müntz-Szász theorem",
    ],
    project_urls={
        "Paper": "https://arxiv.org/abs/2024.XXXXX",
        "Bug Reports": "https://github.com/[username]/muntz-szasz-networks/issues",
        "Source": "https://github.com/[username]/muntz-szasz-networks",
    },
)
