"""
Setup script for RL Lab framework

This installs the framework in development mode so you can
import modules and run experiments.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements (only uncommented lines)
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                requirements.append(line)

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
        'all': [
            'wandb>=0.15.0',
            'optuna>=3.2.0',
            'dm-control>=1.0.14',
            'opencv-python>=4.8.0',
            'pillow>=10.0.0',
            'imageio>=2.31.0',
            'seaborn>=0.12.0',
            'scikit-learn>=1.3.0',
            'flake8>=6.0.0',
            'mypy>=1.4.0',
        ],
        'envs': [
            'dm-control>=1.0.14',
            'atari-py>=0.2.9',
            'mujoco>=2.3.0',
            'box2d-py>=2.3.5', 
            'pybullet>=3.2.5',
        ],
        'dev': [
            'wandb>=0.15.0',
            'optuna>=3.2.0',
            'flake8>=6.0.0',
            'mypy>=1.4.0',
        ],
        'vision': [
            'opencv-python>=4.8.0',
            'pillow>=10.0.0',
            'imageio>=2.31.0',
        ],
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