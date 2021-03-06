from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='SymSim',
      version='0.01',
      description='Tool for simulating planes of symmetries assuming kinematic diffraction',
      long_description=readme(),
      keywords='Simulations STEM Electron Microscopy Glass',
      url='https://github.com/CSSFrancis/SymSim',
      author='CSSFrancis',
      author_email='csfrancis@wisc.edu',
      liscense='MIT',
      packages=['SymSim',
                'SymSim.utils',
                'SymSim.draw',
                'SymSim.sim'],
      install_requires=['hyperspy >=1.5',
                        'numpy>=1.10,!=1.70.0',
                        'matplotlib',
                        'scipy',
                        'scikit-image'],
      zip_safe=False)