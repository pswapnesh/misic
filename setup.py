import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MiSiC", # Replace with your own username
    version="1.0.8",
    author="S.Panigrahi",
    author_email="spanigrahi@imm.cnrs.fr",
    description="Microbe segmentation in dense colonies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://imm.cnrs.fr",
    packages=setuptools.find_packages(),
    package_data={'': ['MiSiDC04082020.h5']},
    include_package_data=True,
    install_requires=[
   'h5py==2.10.0',
   'scikit-image',
   'tensorflow==2.1.0',
   'tqdm'
    ],
    entry_points = {
        'console_scripts': ['mbnet=MiSiC.misic_main:main'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
