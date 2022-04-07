# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = ../docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

build_docs:
	rm -rf source
	cp -rf manual_source source
	make build_guides
	sphinx-apidoc -o source/ ../relevanceai && make clean && make html

build_guides:
	find manual_source/guides/ -type f -name *guide*.rst -delete
	jupyter nbconvert --to rst ../guides/*guide*.ipynb --output-dir=manual_source/guides/

# mv ../guides/*guide*.rst manual_source/guides/
# ls -d -- ../guides/*/ | xargs -I '{}' mv -f {} manual_source/guides/
