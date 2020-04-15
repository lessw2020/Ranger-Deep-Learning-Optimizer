#!/usr/bin/env python

import os
from setuptools import find_packages, setup


setup(
    name='ranger',
    version='0.0.1',
    packages=find_packages(
        exclude=['tests', '*.tests', '*.tests.*', 'tests.*']
    ),
    package_dir={'ranger': os.path.join('.', 'ranger')},
    description='Ranger - a synergistic optimizer using RAdam '
                '(Rectified Adam) and LookAhead in one codebase ',
    author='Less Wright',
    license='Apache',
    install_requires=['torch']
)
