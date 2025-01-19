#!/bin/bash

pandoc NOTE.md -o NOTE.pdf \
  --pdf-engine=xelatex \
  -V geometry:"margin=1in" \
  -V documentclass=article \
  -V mainfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono" \
  -V mathfont="DejaVu Math TeX Gyre" \
  --highlight-style=tango \
  -f markdown+raw_html \
  --wrap=preserve \
  --extract-media=./extracted-media \
  -V colorlinks=true \
  --standalone