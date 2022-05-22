#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/3 19:45
"""

import setuptools
import os

with open("/home/projects/TFpackageText/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

pathToRequiredLib = "/home/projects/TFpackageText"

def getRequiredLib(path):
    with open(os.path.join(path, "requirements.txt"), "r", encoding="utf-8") as f:
        if f.readline().split()[0] == "Package":
            processRequiredLib(path)
        return readRequiredLib(path)

def processRequiredLib(path):
    with open(os.path.join(path, "requirements.txt"), "r", encoding="utf-8") as f:
        w = open(os.path.join(path, "requirements_1.txt"), "w", encoding="utf-8")
        for n, line in enumerate(f):
            if n==0 or n==1:
                continue
            line = line.strip()
            line = line.split()
            line = "==".join(line)
            w.write(line + "\n")
        w.close()
    os.remove(os.path.join(path, "requirements.txt"))
    os.rename(os.path.join(path, "requirements_1.txt"), os.path.join(path, "requirements.txt"))

def readRequiredLib(path):
    with open(path + "/requirements.txt", "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            lines.append(line.strip())
    return lines


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
    install_requires=getRequiredLib(pathToRequiredLib),
    tests_require=[
        'pytest>=3.3.1',
        'pytest-cov>=2.5.1',
    ],
    package_data={'': ['textMG/main_multiGPU.py']},
    python_requires=">=3.7",
)