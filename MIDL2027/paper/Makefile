main.pdf: main.tex sections/*.tex ../references.bib
	pdflatex -interaction=nonstopmode main
	bibtex main
	pdflatex -interaction=nonstopmode main
	pdflatex -interaction=nonstopmode main

clean:
	rm -f main.aux main.bbl main.blg main.log main.out
