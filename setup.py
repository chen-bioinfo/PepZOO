from setuptools import find_packages
from setuptools import setup

setup(
    name='HydrAMP',
    version='1.2.0',
    description='Python package for peptide generation',
    author='Paulina Szymczak',
    author_email='szymczak.pau@gmail.com',
    url='https://hydramp.mimuw.edu/',
    packages=find_packages(),
    install_requires=[
        'torch~=2.0.0',
        'tensorflow~=2.2.1',
        'tensorflow-probability~=0.10.0',
        'Keras~=2.3.1',
        'Keras-Applications~=1.0.8',
        'Keras-Preprocessing~=1.1.2',
        'cloudpickle~=1.4.1',
        'numpy~=1.18.5',
        'pandas~=1.1.5',
        'scikit-learn~=1.2.2',
        'modlamp~=4.2.3',
        'matplotlib~=3.3.4',
        'protobuf~=3.14.0',
        'seaborn~=0.11.2',
        'setuptools~=58.0.0',
        'joblib~=1.3.2',
        'argparse',
        'tqdm~=4.51.0',
        'torchsnooper',
        'Bio',
        'Levenshtein'
    ],
    setup_requires=['wheel']
)
