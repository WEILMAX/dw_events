#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

requirements = ['pandas', 'scikit-learn',  'dataclasses']
test_requirements = ['pytest>=3', ]

setup(
    name="dw_events",
    author_email='maximillian.f.weil@gmail.com',
    python_requires='>=3.6',
    packages=find_packages(include=['dw_events', 'dw_events.*']),
    install_requires=requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/WEILMAX/dw_events',
    version="0.1.0",
    description="Data analytics to detect events in structural data.",
    author="Maximillian Weil",
    license="MIT",
)
