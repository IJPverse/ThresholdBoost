from setuptools import setup, find_packages

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="thresholdboost", # The name users will pip install
    version="0.1.0",       # Initial release version
    author="Israt Jahan Powsi, Rayhan Miah, Md Khorshed Alam",    # <--- PUT YOUR NAME HERE
    author_email="ijahan23.phy@bu.ac.bd", # <--- PUT YOUR EMAIL HERE
    description="A custom ensemble machine learning package for binary classification.",
    long_description=long_description,
    url="https://github.com/IJPverse/ThresholdBoost", # <--- YOUR GITHUB LINK
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
)
