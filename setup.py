
from setuptools import setup,find_packages
with open("requirements.txt") as f:
    Requirements=f.read().splitlines()

    setup(
        name="Smart Home Energy Management",
        version="0.1",
        author="K.Vardhan",
        packages=find_packages(),
        install_requires=Requirements,
    )
