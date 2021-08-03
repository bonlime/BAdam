import re
from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = f.read().split()

with open("badam/__init__.py") as f:
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    version = re.search(VSRE, f.read()).group(1)

setup(
    name="badam",
    version=version,
    author="Emil Zakirov",
    author_email="bonlimezak@gmail.com",
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    url="https://github.com/bonlime/BAdam",
    description="B(onlime)Adam with modifications",
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    setup_requires=["setuptools_scm"],
    python_requires=">=3, <4",
    install_requires=requirements,
    license="MIT License",
)
