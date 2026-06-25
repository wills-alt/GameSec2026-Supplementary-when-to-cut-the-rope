from setuptools import setup, find_packages

setup(
    name="Bayesian_timing_game",
    version="0.1.0",
    author="Willie Kouam",
    author_email="kouamwillie00@gmail.com",
    description="First attempt to merge CTR, Bayesian Optimization and LIPO+ as python library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="  ",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: ",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "networkx>=3.2.1",  
        "numpy>=2.0.0",     
        "scipy>=1.12.0",    
        "matplotlib>=3.8.0" 
    ],
    include_package_data=True,
)
