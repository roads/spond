# Spond: A python library for aligning conceptual systems.

## What's in a name?
The name `Spond` is short for correspondence. The alogorithms in this package seeek to identify a correspondence between different conceptual systems.

## Purpose


## Installation

There is not yet a stable version (nor an official release of this library). All APIs are subject to change and all releases are alpha.

To install the latest development version, clone from GitHub and instal the local repo using pip.
1. Use `git` to clone the latest version to your local machine: `git clone https://github.com/roads/spond.git`
2. Use `make install` to install the cloned repo which executes `pip install -e` in editable mode. By using editable mode, you can easily update your local copy by use `git pull origin master` inside your local copy of the repo. You do not have to re-install with `pip`.

The package can also be obtained by:
* Manually downloading the latest version at https://github.com/roads/spond.git

**Note:** Do not use `pip` to install, since the version on PyPI is incomplete.

## Quick Start
TODO

## Modules
TODO

## Style Checks and Formatting
To check if your code complies with `pycodestyle` (PEP 8) and `pydocstyle` (PEP 257), use
```bash
make flake8
```
To automatically reformat your code for `pycodestyle` and sort import stastements, use
```bash
make lint
```
which runs `black` for formatting and `isort` for sorting. Make sure that the dependent libraries are installed by running
```bash
make install
```

## Contributers
PI:
* Prof. Bradley C. Love

Project Lead:
* Dr. Brett D. Roads

Research Scientist:
* Kaarina Aho
* Kengo Arao
* Roseline Polle

Please see the list of contributors for a complete list of who participated in this project. If you would like to contribute, please see [CONTRIBUTING][CONTRIBUTING.md] for additional guidance.

## Licence
This project is licensed under the Apache Licence 2.0 - see LICENSE file for details.

## Code of Conduct
This project uses a Code of Conduct [CODE](CODE.md) adapted from the [Contributor Covenant][homepage], version 2.0, available at <https://www.contributor-covenant.org/version/2/0/code_of_conduct.html>.