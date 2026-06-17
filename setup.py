"""
TribeBot T9 — package setup
Install in editable mode:  pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="tribebot-t9",
    version="0.9.0",
    description="TribeBot T9 — Advanced Reasoning Large Language Model (research)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="TribeBot Research Team",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.38.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "einops>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "flash": ["flash-attn>=2.5.0"],
        "wandb": ["wandb>=0.16.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
