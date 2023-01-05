#from distutils.core import setup
from __future__ import with_statement
from setuptools import setup, find_packages
import os

VERSION = '0.0.1'
DESCRIPTION = 'Intuitive sound synthesis in python.'

def parse_requirements(file):
    required_packages = []
    with open(os.path.join(os.path.dirname(__file__), file)) as req_file:
        for line in req_file:
            required_packages.append(line.strip())
    return required_packages

def parse_readme(file):
    with open(os.path.join(os.path.dirname(__file__), file)) as rm_file:
        return rm_file.read()

setup(
    name='pynth',
    version=VERSION,
    description=DESCRIPTION,
    author='Matej Bevec',
    author_email='matejbevec98@gmail.com',
    url='https://github.com/MatejBevec/pynth',
    long_description=parse_readme("README.md"),
    long_description_content_tpye="text/markdown",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    keywords=['synth', 'modular', 'sound', 'music', 'computer music', 'unit generators', 'synthesis', 'intuitive', 'python'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers, artists, enthusiasts",
        "Programming Language :: Python :: 3"
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Unix",
        "Operating System :: POSIX"]
)