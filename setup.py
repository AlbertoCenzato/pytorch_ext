import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch_ext",
    version="0.10.0",
    author="Alberto Cenzato",
    author_email="alberto.cenzato@outlook.it",
    description="Extension modules for Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlbertoCenzato/pytorch_ext",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows :: Windows 10",
    ],
)