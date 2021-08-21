from setuptools import setup, find_packages

setup(
    name='wikie',
    packages=find_packages(),
    install_requires=[
        'stanza',
        'pandas'
    ]
)