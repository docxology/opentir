"""
Setup configuration for Opentir package.
A comprehensive toolkit for working with Palantir's open source ecosystem.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="opentir",
    version="1.0.0",
    author="Opentir Contributors",
    author_email="contributors@opentir.org",
    description="A comprehensive toolkit for working with Palantir's open source ecosystem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/opentir",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Version Control :: Git",
        "Topic :: Documentation",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'opentir=src.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 