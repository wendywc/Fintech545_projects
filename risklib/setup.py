from setuptools import setup

setup(
    name = 'risklib',
    version = '0.0.1',
    author = 'Wenjie Cui',
    description='Quatitative risk management package',
    packages =['risklib'],
    install_requires=['pandas','numpy','scipy'],
)