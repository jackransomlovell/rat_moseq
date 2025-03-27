from setuptools import setup, find_packages

setup(
    name="rat-moseq",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "h5py>=3.0.0",
        "opencv-python>=4.5.0",
    ],
    author="Harvard Medical School",
    description="A package for analyzing rat motion sequences",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
) 