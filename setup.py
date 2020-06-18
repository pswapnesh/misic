import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="microbeNet", # Replace with your own username
    version="1.0.0",
    author="S.Panigrahi",
    author_email="spanigrahi@imm.cnrs.fr",
    description="Segmentation of bacteria like objects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://imm.cnrs.fr",
    packages=setuptools.find_packages(),
    #packages=[''],
    install_requires=[
   'scikit-image',
   'tensorflow==2.0.1',
   'tqdm'
    ],
    entry_points = {
        'console_scripts': ['mbnet=mbnet.mbnet:main'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)