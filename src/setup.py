from setuptools import setup, find_packages

setup(
    name='wikie',
    packages=find_packages(),
    install_requires=[
        'stanza',
        'pandas==1.2.3',
        'torch==1.8.1',
        'numpy==1.20.1',
        'sklearn',
        'transformers==4.6.1'
    ]
)