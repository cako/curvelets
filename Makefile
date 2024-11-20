UV := $(shell command -v uv 2> /dev/null || command which uv 2> /dev/null)

.PHONY: test

uvcheck:
ifndef UV
	$(error "Ensure uv is in your PATH")
endif
	@echo Using uv: $(UV)


test39:
	make uvcheck
	$(UV) tool run nox -s tests-3.9

test310:
	make uvcheck
	$(UV) tool run nox -s tests-3.10

test311:
	make uvcheck
	$(UV) tool run nox -s tests-3.11

test312:
	make uvcheck
	$(UV) tool run nox -s tests-3.12

test313:
	make uvcheck
	$(UV) tool run nox -s tests-3.13

test:
	make test39
	make test310
	make test311
	make test312
	make test313

lint:
	make uvcheck
	$(UV) tool run nox -s lint

pylint:
	make uvcheck
	$(UV) tool run nox -s pylint

docs:
	make uvcheck
	$(UV) tool run nox -s docs

servedocs:
	make uvcheck
	$(UV) tool run nox -s docs -- --serve --port 1234

precommitupdate:
	make uvcheck
	$(UV) tool run pre-commit autoupdate
