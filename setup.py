#!/usr/bin/env python3

from setuptools import setup, find_packages


setup(
    name="hts",
    version="DEV",
    license="GPL3",
    description="Read file in reverse order",
    author="Antonis Christofides",
    author_email="antonis@antonischristofides.com",
    url="https://github.com/openmeteo/hts",
    packages=find_packages(),
    install_requires=['pandas>=0.14'],
    test_suite="tests",
    tests_require=['iso8601'],
)
