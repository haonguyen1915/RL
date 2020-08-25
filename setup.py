import os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_packages = f.readlines()

setup(
    name='rl',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    py_modules=['rl'],
    install_requires=required_packages,
    python_requires='>3.6.0',
    package_data={
    },

    entry_points={
        'console_scripts': [
            'rl = rl.run_cli:entry_point'
        ]
    },
)
