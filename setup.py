"""Setup file."""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='spond',
    version='0.1.0',
    description='Toolbox for aligning conceptual systems.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    author='Brett D. Roads',
    author_email='brett.roads@gmail.com',
    license='Apache Licence 2.0',
    packages=['spond'],
    python_requires='>=3.8, <3.9',
    install_requires=['torch', 'torchvision', 'numpy'],
    include_package_data=True,
    url='https://github.com/roads/spond',
    download_url=''
)
