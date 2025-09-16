# The Little Book of Algorithms

A concise, beginner-friendly introduction to Algorithms.

```
```

## Formats

- [Download PDF](docs/book.pdf) — print-ready
- [Download EPUB](docs/book.epub) — e-reader friendly
- [View LaTeX](docs/book-latex/book.tex) — `.tex` source
- [Read on GitHub Pages](https://little-book-of.github.io/algorithms/) — online website

## Build it yourself (Quarto)

Install Quarto: [https://quarto.org/docs/get-started/](https://quarto.org/docs/get-started/)

Preview locally (auto-reload):

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

All contributions are welcome—small fixes, new examples, clarifications, figures, exercises, or larger structural improvements.

**How to contribute**

1. Open an issue to describe the change (or reference an existing one).
2. Fork the repository and create a branch for your work.
3. Keep commits focused, with short explanations in messages or the PR description.
4. Submit a pull request; include the section(s) you changed and a brief rationale.

**Style guidelines (brief)**

- Prefer short paragraphs and plain language.
- Keep notation consistent with nearby sections.
- Include minimal, self-contained examples where helpful.
- Use LaTeX/MathJax for math; keep equations readable.

## Citation

Nguyen, Duc-Tam (2025). *The Little Book of Algorithms* 

```
@book{Nguyen2025TLBoAlgo,
  author = {Duc-Tam Nguyen},
  title  = {The Little Book of Algorithms},
  year   = {2025},
  url    = {https://github.com/little-book-of/algorithms}
}
```

## License

This work is licensed under **CC BY-NC-SA 4.0**. See [LICENSE](LICENSE) for the full text.
