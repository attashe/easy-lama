# setup.py
from setuptools import setup, find_packages

setup(
    name="easy_lama",
    version="1.0.0",
    description="Streamlined LAMA (Large Mask Inpainting) inference library",
    long_description="A clean, minimal implementation of LAMA inpainting with simple API",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/easy-lama",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "numpy",
        "Pillow",
        "omegaconf",
        "PyYAML",
        "safetensors>=0.3.0",
    ],
    entry_points={
        'console_scripts': [
            'lama-inpaint=easy_lama.cli:main',
            'lama-convert=easy_lama.convert:main',
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)