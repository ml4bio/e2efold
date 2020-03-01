from setuptools import setup

import os

BASEPATH = os.path.dirname(os.path.abspath(__file__))

setup(name='e2efold',
      py_modules=['e2efold'],
      install_requires=[
          'torch'
      ],
)
