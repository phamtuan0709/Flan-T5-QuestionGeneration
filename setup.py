"""
Setup script for Bloom-QG package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="bloom-qg",
    version="1.0.0",
    author="Bloom-QG Team",
    description="Bloom-Controlled Question Generation with Hybrid Neural Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="question-generation bloom-taxonomy transformers nlp",
    entry_points={
        'console_scripts': [
            'bloom-qg-train=bloom_qg.train_gpu:main',
            'bloom-qg-infer=bloom_qg.test_local:main',
            'bloom-qg-eval=bloom_qg.evaluate:main',
            'bloom-qg-prepare=bloom_qg.data.prepare_squad:main',
        ],
    },
)
