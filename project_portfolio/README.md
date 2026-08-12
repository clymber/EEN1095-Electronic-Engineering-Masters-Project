# EEN1095 Main Research Paper

LaTeX source for the six-page IEEE Transactions-style paper that forms the
centrepiece of the EEN1095 project portfolio.

## Project structure

```text
.
├── main.tex
├── references.bib
├── IEEEtran.cls
├── appendices/
│   ├── appendix_b_status_report.tex
│   ├── appendix_c_project_plan_progress_achievements.tex
│   ├── appendix_d_design_and_implementation_details.tex
│   └── appendix_e_testing_and_supplementary_results.tex
├── documents/
│   ├── ChunyuLong_A00049113_StatusReport.pdf
│   └── ProjectPlan_A00049113_v4.0.pdf
├── evidence/
│   └── README.md
├── sections/
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_methodology.tex
│   ├── 04_results.tex
│   ├── 05_analysis.tex
│   └── 06_conclusion.tex
└── media/
    ├── appendix_d/
    └── README.md
```

`main.tex` builds only the six-page paper. The Appendix B--E fragments remain
outside that page limit until the full portfolio wrapper is assembled.
Appendices B and C import unchanged EEN1101 PDFs and require the future wrapper
to load `pdfpages`. Appendix C also requires `booktabs` and provides page-style
hooks for the imported plan and its C.2--C.5 content. NB07 regenerates the
machine-readable testing evidence, while NB08 regenerates the selected
neural-model diagrams and inventory documented in `evidence/README.md`.

## Overleaf

1. Create a blank Overleaf project.
2. Upload this complete folder or upload its ZIP archive.
3. Set `main.tex` as the main document if Overleaf does not detect it
   automatically.
4. Use the pdfLaTeX compiler.

The project has no absolute paths, shell commands, or local-only dependencies.
The root `IEEEtran.cls` is synchronised with the supplied Transactions template
in `documents/IEEE-Transactions-LaTeX2e-templates-and-instructions/`; keep that
copy with the source when uploading to Overleaf.

## Local build

```bash
latexmk -pdf main.tex
```

The project-level `.latexmkrc` sets the generated PDF filename to
`A00049113_EEN1095_Project_Portfolio.pdf`. The same filename is configured for
LaTeX Workshop builds in VS Code. Both workflows write the PDF, auxiliary files
and SyncTeX data to `build/`; avoid invoking `pdflatex` directly without an
explicit output directory.

Clean generated files with:

```bash
latexmk -C
```

## Portfolio constraints reflected in the scaffold

- A4 paper size.
- IEEEtran two-column journal layout using the supplied Transactions template.
- Native IEEE running heads with `EEN1095 Project Portfolio`, `August 2026`,
  a short author/title head, and main-paper page numbering.
- Separate source files for each paper section.
- BibTeX references using the IEEE bibliography style.
- All figures stored under `media/`.
- Comments indicating the recommended six-page allocation.
