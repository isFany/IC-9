from setuptools import setup,find_packages
import os
import sys



with open('requirements.txt') as f:
    # Separate by lines ('\r','\r\n', \n'), 
    # and return a list containing each line as an element,
    required = f.read().splitlines()


reqs = []
for element in required:
    reqs +=[element]


setup(
    name = 'ACSE9',
    version ='1.0',
    description = 'A comparison of dimensionality reduction methods '
                 +' for fluid flow problems focusing on hierarchical autoencoders',
    author='Fan Yang',
    author_email='fan.yang20@imperial.ac.uk',
    install_requires=reqs,
    test_suite='tests',
    # packages=['fpc_methods']
)