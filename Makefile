.PHONY: setup
setup:
	pipenv sync --dev
	cp .env.template .env
	pipenv run pre-commit install
	pipenv run pre-commit run --all-files
	pipenv run pytest


.PHONY: precommit
precommit:
	pipenv run pre-commit run --all-files
