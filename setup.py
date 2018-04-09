from setuptools import find_packages, setup

setup(name='ensemble_experiments',
      version='0.0.0',
      description='Experimentation with ANN Ensembles',
      author='Alisdair Roberson',
      author_email='u3142846@uni.canberra.edu.au',
      packages=find_packages(),
      entry_points={
        'console_scripts': ['ee = ensemble_experiments:main']}
     )
