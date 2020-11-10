.DEFAULT_GOAL := help

.PHONY: install
install: # Install spond from local using pip, including dependencies
	pip install -e .

.PHONY: lint
lint: ## Run black formatter and isort your imports with isort
	black --quiet -l 120 spond tests examples
	isort .

.PHONY: flake8
flake8: # Check coding style according to pycodestyle (PEP8) and pydocstyle (PEP257). Ignore unused imports.
	flake8 --show-source --max-line-length=120 --ignore=F401

# ref: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help
help: ## Generate this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-45s\033[0m %s\n", $$1, $$2}'	