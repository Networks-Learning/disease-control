from setuptools import setup, find_packages

setup(
    name='disease-control',
    version='0.1',
    url='https://github.com/Networks-Learning/disease-control.git',
    description='Optimal Stochastic Control algorithm for epidemic models',
    packages=find_packages(),
    install_requires=['numpy >= 1.16.2', 'networkx >= 2.0', 'scipy >= 1.2.1',
                      'lpsolvers >= 0.8.9', 'pandas >= 0.24.1'],
)
