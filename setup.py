#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/3 19:45
"""

import setuptools
with open("/home/projects/tensorflow_practice/tf_pro/tf1/package_text/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = []
with open("/home/projects/tensorflow_practice/tf_pro/tf1/package_text//requirements.txt", "r", encoding="utf-8") as f:
    for n, line in enumerate(f):
        if n==0 or n==1:
            continue
        line = line.strip()
        line = line.split()
        line = "==".join(line)
        install_requires.append(line)

setuptools.setup(
    name="TFpackageText",
    version="0.0.1",
    author="Example Author",
    author_email="victor@bai.com",
    description="this is used as a framework for train/eval/pred through multi GPU environment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/victor/tree/master/tf1/package_text",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": ""},
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    tests_require=[
        'pytest>=3.3.1',
        'pytest-cov>=2.5.1',
    ],
    package_data={'': ['textMG/main_multiGPU.py']},
    python_requires=">=3.7",
)