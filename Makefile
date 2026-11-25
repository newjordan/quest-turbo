PYTHON ?= python3

.PHONY: test syntax

test:
	$(PYTHON) -m pytest -q

syntax:
	@fail=0; \
	for f in env.sh tests/certify.sh scripts/*.sh; do \
	  bash -n "$$f" || fail=1; \
	done; \
	exit $$fail
