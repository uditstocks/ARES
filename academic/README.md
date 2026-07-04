# Academic Materials

This folder contains a formal research paper describing the ARES system.

| File | Description |
|------|-------------|
| [`ARES_paper.tex`](ARES_paper.tex) | Typeset, citation-ready paper (two-column, self-contained LaTeX with an embedded architecture figure, algorithm, tables, and bibliography). |
| [`ARES_paper.md`](ARES_paper.md) | Readable Markdown rendering of the same paper (no LaTeX toolchain required). |
| [`references.bib`](references.bib) | BibTeX bibliography, for readers who prefer a `bibtex`/`biblatex` workflow. |

## Building the PDF

The `.tex` source is **self-contained** — its bibliography is embedded, so no
BibTeX pass is needed. It uses only widely-available packages (TikZ, `booktabs`,
`algorithm`/`algpseudocode`, `listings`, `hyperref`).

### Option A — local LaTeX (TeX Live / MiKTeX)

```bash
cd academic
pdflatex ARES_paper.tex
pdflatex ARES_paper.tex     # second pass resolves cross-refs and TikZ layout
```

The TikZ architecture figure and internal references settle on the **second**
pass, so always run `pdflatex` twice.

Required TeX packages (all in a standard full install):
`geometry`, `lmodern`, `microtype`, `amsmath`, `graphicx`, `booktabs`,
`enumitem`, `xcolor`, `listings`, `algorithm`, `algpseudocode`, `tikz`,
`hyperref`, `caption`.

On a minimal TeX Live, install the collections that provide them:

```bash
# Debian/Ubuntu
sudo apt-get install texlive-latex-recommended texlive-latex-extra \
                     texlive-pictures texlive-science texlive-fonts-recommended
```

### Option B — no local LaTeX

Upload `ARES_paper.tex` (and optionally `references.bib`) to
[Overleaf](https://www.overleaf.com/) and compile with **pdfLaTeX**. It builds
as-is.

### Option C — read the Markdown

If you just want to read the paper, open [`ARES_paper.md`](ARES_paper.md) — it
mirrors the LaTeX content with an ASCII architecture diagram.

## Switching to a BibTeX workflow (optional)

The paper embeds its references for zero-friction builds. To use `references.bib`
instead, replace the `\begin{thebibliography}...\end{thebibliography}` block in
`ARES_paper.tex` with:

```latex
\bibliographystyle{IEEEtran}
\bibliography{references}
```

then build with `pdflatex → bibtex → pdflatex → pdflatex`.

## Citing

```bibtex
@techreport{sharma2026ares,
  title  = {{ARES}: A Stateful, Human-in-the-Loop Multi-Agent Architecture
            for Grounded Autonomous Research Synthesis},
  author = {Sharma, Udit},
  year   = {2026},
  note   = {Technical report}
}
```
