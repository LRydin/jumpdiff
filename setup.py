import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jumpdiff",
    version="0.4.2",
    author="Leonardo Rydin Gorjão",
    author_email="leonardo.rydin@gmail.com",
    description="JumpDiff: Non-parametric estimators for jump-diffusion processes for Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LRydin/JumpDiff",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT License",
    python_requires='>=3.5',
)
