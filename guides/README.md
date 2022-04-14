# Guides

To build guides into a references, place the `.ipynb` notebook containing the guide into this directory.

Then run `cd docsrc`

and finally `make build_guides`

this will convert all `.ipynb` files with guide in the title to `.rst` files to be used in references.

This should be run before `make build_docs`



# Guide Template

- Title should be H1 (1 hashtag)
- Subtitles should be H2 (2 hashtags)

Function names should be explicit

Markdown should contain info and not docstrings and users should be reading markdown where possible.
Comments are useful for additional options.
