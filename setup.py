from setuptools import setup, find_packages

setup(
    name='vision-transformer',
    packages=find_packages(exclude=[]),
    version='0.0.1',
    license='MIT',
    description='PyTorch implementation of Vision Transformer paper: https://arxiv.org/abs/2010.11929',
    author='Daniil Korolev',
    url='https://github.com/Danielto1404/vision-transformer',
    keywords=[
        'Artificial Intelligence',
        'OpenAI gym',
        'Deep Learning',
        'Reinforcement Learning',
        'Trading environment',
        'Crypto'
    ],
    install_requires=[
        'numpy',
        'torch'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.9',
    ],
)
