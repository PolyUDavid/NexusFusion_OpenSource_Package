"""
NexusFusion Setup Script
========================

Setup script for the NexusFusion multi-modal fusion architecture.

Authors: NexusFusion Research Team
License: MIT
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Package metadata
setup(
    name="nexusfusion",
    version="1.0.0",
    author="NexusFusion Research Team",
    author_email="nexusfusion-team@example.com",
    description="Multi-Modal Fusion Architecture for Verifiably Safe Cooperative Driving",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nexusfusion/NexusFusion",
    project_urls={
        "Bug Tracker": "https://github.com/nexusfusion/NexusFusion/issues",
        "Documentation": "https://nexusfusion.readthedocs.io",
        "Paper": "https://arxiv.org/abs/2024.xxxxx",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Networking",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "viz": [
            "mayavi>=4.8.0",
            "vtk>=9.1.0",
            "plotly>=5.0.0",
        ],
        "gpu": [
            "torch-geometric>=2.2.0",
            "pyg-lib>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nexusfusion-train=nexusfusion.training.train_nexus_fusion:main",
            "nexusfusion-infer=nexusfusion.inference.nexus_fusion_api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "nexusfusion": [
            "configs/*.json",
            "figures/*.png",
            "figures/*.pdf",
        ],
    },
    keywords=[
        "autonomous driving",
        "multi-modal fusion", 
        "cooperative perception",
        "byzantine fault tolerance",
        "trajectory prediction",
        "v2x communication",
        "graph neural networks",
        "transformer",
        "pytorch",
        "deep learning",
        "computer vision",
        "robotics",
        "safety-critical systems",
    ],
    zip_safe=False,
)
