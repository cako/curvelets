UV := $(shell command -v uv 2> /dev/null || command which uv 2> /dev/null)

.PHONY: test cleandoc

uvcheck:
ifndef UV
	$(error "Ensure uv is in your PATH")
endif
	@echo Using uv: $(UV)


test39:
	make uvcheck
	unset NO_COLOR FORCE_COLOR; $(UV) tool run nox -s tests-3.9

test310:
	make uvcheck
	unset NO_COLOR FORCE_COLOR; $(UV) tool run nox -s tests-3.10

test311:
	make uvcheck
	unset NO_COLOR FORCE_COLOR; $(UV) tool run nox -s tests-3.11

test312:
	make uvcheck
	unset NO_COLOR FORCE_COLOR; $(UV) tool run nox -s tests-3.12

test313:
	make uvcheck
	unset NO_COLOR FORCE_COLOR; $(UV) tool run nox -s tests-3.13

test314:
	make uvcheck
	unset NO_COLOR FORCE_COLOR; $(UV) tool run nox -s tests-3.14

test:
	make test39
	make test310
	make test311
	make test312
	make test313
	make test314

lint:
	make uvcheck
	unset NO_COLOR FORCE_COLOR; $(UV) tool run nox -s lint

pylint:
	make uvcheck
	unset NO_COLOR FORCE_COLOR; $(UV) tool run nox -s pylint

doc:
	make uvcheck
	cd docs && rm -rf source && sphinx-apidoc -f -M -o source/ ../src && cd .. && unset NO_COLOR FORCE_COLOR; $(UV) tool run nox -s docs

cleandoc:
	rm -rf docs/_build docs/source docs/auto_examples

servedocs:
	make uvcheck
	unset NO_COLOR FORCE_COLOR; $(UV) tool run nox -s docs -- --serve --port 1234

watchdoc:
	while inotifywait -q -r src/ examples/ -e create,delete,modify; do { make doc; }; done

precommitupdate:
	make uvcheck
	$(UV) tool run pre-commit autoupdate
