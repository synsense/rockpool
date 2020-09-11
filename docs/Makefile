# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	cp -f $(SOURCEDIR)/../CHANGELOG.md advanced/
	rm -rf $(SOURCEDIR)/reference/_autosummary/*
	find . -name .ipynb_checkpoints -d -print0 | xargs -0 rm -r --
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml:
	sphinx-autobuild \
		--open-browser $(SOURCEDIR) $(BUILDDIR)/html \
		--ignore "*.tmp*" \
		--ignore "*.ai" \
		--ignore "*.tmp" \
		--ignore "*.ipynb_checkpoints/*" \
		--ignore "*.~*" \
		--ignore "*.xml" \
		--ignore "Makefile" \
		--ignore "_build" \
		--ignore "reference/_autosummary" \
		--delay 5 \
		--watch ../rockpool

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


