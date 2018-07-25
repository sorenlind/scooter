"""Setup script for package."""
import re
from setuptools import setup, find_packages

VERSION = re.search(r'^VERSION\s*=\s*"(.*)"', open("scooter/version.py").read(), re.M).group(1)
with open("README.md", "rb") as f:
    LONG_DESCRIPTION = f.read().decode("utf-8")

setup(
    name="scooter",
    include_package_data=True,
    version=VERSION,
    description="Simple RESTful API for serving ML models.",
    long_description=LONG_DESCRIPTION,
    author="Soren Lind Kristiansen",
    author_email="sorenlind@mac.com",
    url="https://github.com/sorenlind/scooter/",
    keywords="machine learning api flask rest",
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'h5py', 'flask', 'flask-cors', 'requests', 'redis'],
    extras_require={
        'test': ['pytest', 'tox'],
        'dev': ['pylint', 'pycodestyle', 'pydocstyle', 'yapf', 'rope'],
        'resnet': ['keras', 'tensorflow', 'Pillow'],
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
    ])
