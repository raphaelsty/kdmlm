import setuptools

from kdmlm.__version__ import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="kdmlm",
    version=f"{__version__}",
    author="",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=required,
    package_data={
        "kdmlm": [
            "datasets/*.txt",
            "datasets/sentences/*.txt",
            "datasets/*.json",
            "datasets/fb15k237one/*.json",
            "datasets/fb15k237one/*.csv",
            "datasets/wiki_fb15k237one/*.json",
            "datasets/wiki_fb15k237_test/*.json",
            "datasets/wiki_fb15k237one_test/*.json",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
