from setuptools import setup

setup(
    name="Surveillance",
    version="1.1.0",
    description="Classes implementing a Surveillance system for the SuperviseIt project",
    author="IVALab",
    packages=["Surveillance"],
    install_requires=[
        "psutil",
        "transitions",
        "python-benedict",
        "Lie @ git+https://github.com/ivapylibs/Lie.git",
        "camera @ git+https://github.com/ivapylibs/camera.git",
        "ROSWrapper @ git+https://github.com/ivaROS/ROSWrapper.git",
        "improcessor @ git+https://github.com/ivapylibs/improcessor.git",
        "trackpointer @ git+https://github.com/ivapylibs/trackpointer.git",
        "detector @ git+https://github.com/ivapylibs/detector.git",
        "perceiver @ git+https://github.com/ivapylibs/perceiver.git",
        # NOTE: is this package necessary?
        "puzzleSolvers @ git+https://github.com/ADCLab/puzzleSolvers.git",
        "puzzle_solver @ git+https://github.com/ivapylibs/puzzle_solver.git",
    ],
)
