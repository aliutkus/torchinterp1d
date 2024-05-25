from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Proceed to setup
setup(
    name='torchinterp1d',
    version='1.1',
    description='An interp1d implementation for pytorch',
    download_url='https://github.com/aliutkus/torchinterp1d/archive/refs/tags/v1.tar.gz',
    url='https://github.com/aliutkus/torchinterp1d',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Antoine Liutkus',
    author_email='antoine.liutkus@inria.fr',
    packages=['torchinterp1d'],
    keywords='interp1d torch',
    install_requires=[
        'torch>=1.6',
    ],
    )
