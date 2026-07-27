.PHONY: all clean

all:
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	BSTINPUTS=../pvas_v2//: bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(RM) main.aux main.bbl main.blg main.brf main.log main.out main.pdf
