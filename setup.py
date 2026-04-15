from setuptools import setup, find_packages

setup(
    name="thresholdboost",
    version="0.1.0",
    author="Your Name",
    description="Gradient boosting using differentiable soft threshold units.",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "scikit-learn"],
)