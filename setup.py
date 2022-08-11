from setuptools import setup
setup(name='Surveillance',
      version='1.0',
      description='Classes implementing a Surveillance system for the SuperviseIt project',
      author='IVALab',
      packages=['Surveillance'],
      install_requires=["psutil", "transitions", "python-benedict>=0.25", "yacs>=0.1.8"]
      )
