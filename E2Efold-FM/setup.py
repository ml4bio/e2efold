from setuptools import setup

import os

BASEPATH = os.path.dirname(os.path.abspath(__file__))

setup(name='e2efoldFM',
      py_modules=['e2efoldFM'],
      install_requires=[
          'torch'
      ],
)
