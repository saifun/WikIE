from setuptools import setup, find_packages

setup(
    name='wikie',
    packages=find_packages(),
    version='1.0.0',
    install_requires=[
        'stanza==1.2',
        'pandas==1.2.3',
        'torch==1.8.1',
        'numpy==1.20.1',
        'sklearn',
        'transformers==4.6.1'
    ]
)