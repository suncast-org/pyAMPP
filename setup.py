#! /usr/bin/env python

from __future__ import absolute_import, division, print_function

from setuptools import setup, Extension

import os, glob, numpy, subprocess, sys

MATPLOTLIB_DEP = 'matplotlib'
ASTROPY_DEP = 'astropy>=3.0'


# def get_description():
#     def get_description_lines():
#         seen_desc = False
#         with open('README.md', encoding='utf-8') as f:
#             for line in f:
#                 if seen_desc:
#                     if line.startswith('##'):
#                         break
#                     line = line.strip()
#                     if line:
#                         yield line
#                 elif line.startswith('## Features'):
#                     seen_desc = True
#
#     return ' '.join(get_description_lines())
#
# print(get_description())


def indir(path, files):
    return [os.path.join(path, f) for f in files]


global_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

# with open('README.md', encoding='utf-8') as f:
#     long_description = f.read()

setup(
    name='pyampp',
    setup_requires=['setuptools_scm'],
    version='0.1.1.2',
    description='automatic model production pipeline (AMPP)',
    # long_description=get_description(),
    long_description='',
    long_description_content_type='text/markdown',
    license_files=("LICENSE", "LICENSE-GPL"),
    author='The SUNCAST team',
    author_email='sijie.yu@njit.edu',
    url='https://github.com/suncast-org/pyAMPP',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    install_requires=[
        ASTROPY_DEP,
        MATPLOTLIB_DEP,
        'numpy>=1.2',
        'ephem>=3.7.3.2',
        'scipy>=0.19',
        'sunpy[all]>=5.0'
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov'
        ]
    },
    package_dir={'pyampp': 'pyampp', 'pyampp._src': 'pyampp/_src'},
    packages=['pyampp', 'pyampp._src'],
    ext_modules=[
    ],
    scripts=glob.glob('scripts/*'),

    include_package_data=True,
    zip_safe=False,
    test_suite="tests.pyampp_test.TestSuite",
)
