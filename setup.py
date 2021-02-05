import os
from setuptools import setup, find_packages

from pyESI import __version__

here = os.path.abspath(os.path.dirname(__file__))

# Read README.md to use it as the long description
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    reqs = f.read().split('\n')

reqs = [x.strip() for x in reqs if x.strip() != '']

setup(
    name="pyESI",
    version=__version__,
    author="ALGES Lab",
    author_email="contacto@alges.cl",
    description=("Ensemble Spatial Interpolation module for Python"),
    license="MIT",
    keywords="esi, esi-gok, esi-idw, ensemble spatial interpolation",
    url="",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type = "text/markdown",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=reqs,
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ]
)
