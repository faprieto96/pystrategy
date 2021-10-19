import setuptools
from setuptools import setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setuptools.setup(
    name='pystrategy',
    version="0.0.5",
    author = 'Francisco A. Prieto Rodriguez, Francisco de Asís Fernández Navarro, David Becerra Alonso',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = 'https://github.com/faprieto96/pystrategy',
    packages = setuptools.find_packages(),
    classifiers= [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.0',
)