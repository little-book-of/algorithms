# The Little Book of algorithms

## Formats

- [Download PDF](releases/book.pdf) - print-ready
- [Download EPUB](releases/book.epub) - e-reader friendly
- [View LaTeX](releases/book.tex) - `.tex` source
- [Read on GitHub Pages](https://little-book-of.github.io/algorithms/) - online website

## Build it yourself (Quarto)

We use [Quarto](https://quarto.org/docs/get-started/) to generate all outputs.

Preview locally:

```bash
quarto preview
```

Render outputs:

```bash
# All configured formats
quarto render

# Individual formats
quarto render --to html     # site into docs/
quarto render --to pdf      # docs/book.pdf
quarto render --to epub     # docs/book.epub
quarto render --to latex    # docs/book-latex/book.tex
```

## Contributing

We welcome contributions: fixing typos, clarifying sections, adding diagrams, or improving examples.

## Citation

Nguyen, Duc-Tam (2025). _The Little Book of algorithms_

```
@book{Nguyen2025TLBoAlg,
  author = {Duc-Tam Nguyen},
  title  = {The Little Book of algorithms},
  year   = {2025},
  url    = {https://github.com/little-book-of/algorithms}
}
```

## License

Licensed under **CC BY-NC-SA 4.0**. See [LICENSE](LICENSE) for details.
