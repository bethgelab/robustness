import setuptools

#with open("README.md", "r", encoding="utf-8") as fh:
#    long_description = fh.read()

setuptools.setup(
    name="robusta",
    version="0.0.1",
    author="Evgenia Rusak, Steffen Schneider, George Pachitariu",
    author_email="steffen@bethgelab.org",
    description="A pytorch package for robustness and adaptation research",
    long_description="# A pytorch package for robustness and adaptation research ☕",
    long_description_content_type="text/markdown",
    url="https://github.com/bethgelab/robustness",
    project_urls={
        "Bug Tracker": "https://github.com/bethgelab/robustness/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "robusta"},
    packages=setuptools.find_packages(where="robusta"),
    python_requires=">=3.6",
)