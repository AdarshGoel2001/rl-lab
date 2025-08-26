"""
Setup script for RL Lab framework

This installs the framework in development mode so you can
import modules and run experiments.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read description
readme_path = Path(__file__).parent / "CLAUDE.md"
if readme_path.exists():
    with open(readme_path, encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "Modular reinforcement learning research framework"

setup(
    name="rl-lab",
    version="0.1.0",
    description="Modular reinforcement learning research framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="RL Researcher",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'atari': ['atari-py>=0.2.9'],
        'mujoco': ['mujoco>=2.3.0'],
        'box2d': ['box2d-py>=2.3.5'],
        'pybullet': ['pybullet>=3.2.5'],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)