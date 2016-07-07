#!/usr/bin/env python3

from setuptools import setup, find_packages


setup(
    name="pd2hts",
    version="DEV",
    license="GPL3",
    description=("Read and write Pandas objects from/to Hydrognomon "
                 "timeseries files"),
    author="Antonis Christofides",
    author_email="antonis@antonischristofides.com",
    url="https://github.com/openmeteo/pd2hts",
    packages=find_packages(),
    install_requires=['pandas>=0.14',
                      'textbisect'
                      ],
    test_suite="tests",
    tests_require=['iso8601'],
    classifiers=[
        "Development Status :: 4 - Beta",
        ("License :: OSI Approved :: GNU General Public License "
         "v3 or later (GPLv3+)"),
         'Topic :: Scientific/Engineering",
    ],
)
