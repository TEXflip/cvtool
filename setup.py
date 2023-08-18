import setuptools

# load README.md for long description
# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()
long_description = "Personal handy CV tools"


# load requirements
with open("requirements/base.txt", "r", encoding="utf-8") as fh:
    base_requirements = fh.read().splitlines()

setuptools.setup(
    name="cvtools",
    version="1.0.0",
    author="Michele Tessari",
    author_email="michele.tessari.wk@gmail.com",
    description="Personal handy CV tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["cvtool"],
    install_requires=base_requirements,
    entry_points={
        'console_scripts': [
            'cvtools=cvtools:run'
        ]
    },
)