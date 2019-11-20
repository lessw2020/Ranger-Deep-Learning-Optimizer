from setuptools import setup

setup(
    name='ranger',
    version='0.0.1',
    description='Ranger - a synergistic optimizer using RAdam '
                '(Rectified Adam) and LookAhead in one codebase ',
    author='Less Wright',
    license='Apache',
    install_requires=['torch']
)
