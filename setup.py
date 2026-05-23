from setuptools import setup, find_packages

setup(
    name="Surveillance",
    version="1.1.0",
    description="Classes implementing a Surveillance system for the SuperviseIt project",
    author="IVALab",
    packages=find_packages(),
    install_requires=[
        "psutil",
        "transitions",
        "python-benedict",
        "yacs",
    ],
)
