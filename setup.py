"""!
@defgroup   Surveillance

@brief      Specialized routines for performing surveillance type processing.

Surveillance in this case means to observe a given area or scene and to recognize
objects or activities occuring within the scene.  It may also include
interpretation of those activities as part of a monitoring system.
"""
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
        "Lie @ git+https://github.com/ivapylibs/Lie.git",
        "camera @ git+https://github.com/ivapylibs/camera.git",
        "ROSWrapper @ git+https://github.com/ivaROS/ROSWrapper.git",
        "improcessor @ git+https://github.com/ivapylibs/improcessor.git",
        "trackpointer @ git+https://github.com/ivapylibs/trackpointer.git",
        "detector @ git+https://github.com/ivapylibs/detector.git",
        "perceiver @ git+https://github.com/ivapylibs/perceiver.git",
        # NOTE: is this package necessary?
        "puzzleSolvers @ git+https://github.com/ADCLab/puzzleSolvers.git",
        "puzzle @ git+https://github.com/ivapylibs/puzzle_solver.git",
    ],
)
