# EEN1095 Project Portfolio

LaTeX source for the complete EEN1095 project portfolio, including its
six-page IEEE Transactions-style research paper and Appendices A--F.

## Project structure

```text
.
├── latexmkrc
├── main.tex
├── research_paper.tex
├── appendix_e.tex
├── references.bib
├── IEEEtran.cls
├── appendices/
│   ├── appendix_a_use_of_genai.tex
│   ├── appendix_b_status_report.tex
│   ├── appendix_c_project_plan_progress_achievements.tex
│   ├── appendix_d_design_and_implementation_details.tex
│   ├── appendix_e_testing_and_supplementary_results.tex
│   └── appendix_f_project_repository.tex
├── documents/
│   ├── EEN1095_Biweekly_GenAI_Log.pdf
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

`main.tex` assembles the cover, Acknowledgements, Contents, six-page paper and
Appendices A--F. `research_paper.tex` builds only the six-page paper. Appendix A
imports the PDF export of the cumulative GenAI log; Appendices B and C import
the unchanged EEN1101 PDFs. `appendix_e.tex` builds Appendix E separately so
its full-width IEEE tables settle before the content is imported by the overall
wrapper. NB07 regenerates the testing evidence, while NB08 regenerates the
selected neural-model diagrams and inventory documented in
`evidence/README.md`.

## Overleaf

1. Recreate the upload archive from this directory with:

   ```bash
   cd project_portfolio

   zip -r -FS ../A00049113_EEN1095_Overleaf.zip . \
     -x '.DS_Store' '*/.DS_Store' \
        '.latexmkrc' '.vscode/*' 'build/*' \
        'environment.yml' 'six_page_research_paper_plan.md' \
        'evidence/*' 'media/README.md' \
        'media/appendix_d/*.png' 'media/s7_1_*.png' \
        'documents/EEN1095_Final_Project_Portfolio_Guide.pdf' \
        'documents/EEN1095_Portfolio_Marking_Rubric.pdf' \
        'documents/IEEE-Transactions-LaTeX2e-templates-and-instructions/*'
   ```

   The archive is created one directory above `project_portfolio`, so its root
   contains `main.tex` and `latexmkrc` rather than an enclosing folder. `-FS`
   also removes stale entries when an existing archive is regenerated.
2. Create an Overleaf project by uploading that ZIP archive.
3. Overleaf should select `main.tex` automatically; confirm it is the main
   document if necessary.
4. Use the pdfLaTeX compiler.
5. Select **Recompile**. The root `latexmkrc` automatically builds the
   six-page paper and Appendix E with distinct job names before assembling the
   complete portfolio.

The first compilation builds three documents and therefore takes longer than a
normal incremental compilation. The Appendix D diagrams use lossless PDF
wrappers of the notebook-generated PNG files to keep compilation within
Overleaf's time limit. The project has no absolute paths or local-only
dependencies. The root `IEEEtran.cls` is the supplied Transactions class used
by all three compiled documents; keep it with the source when uploading to
Overleaf.

## Local build

Build the six-page paper:

```bash
latexmk -pdf -jobname=A00049113_EEN1095_Research_Paper research_paper.tex
```

The project-level `.latexmkrc` writes the PDF, auxiliary files and SyncTeX data
to `build/`; avoid invoking `pdflatex` directly without an explicit output
directory.

If the cumulative GenAI log changes, export its current Word source first:

```bash
pandoc ../Documents/EEN1095_Biweekly_GenAI_Log.docx \
  --pdf-engine=xelatex -V papersize:a4 -V geometry:margin=18mm \
  -V pagestyle=empty -o documents/EEN1095_Biweekly_GenAI_Log.pdf
```

Then build the complete portfolio with one command. The `latexmkrc` prerequisite
rules build the paper and Appendix E automatically:

```bash
latexmk -pdf -jobname=A00049113_EEN1095_Full_Project_Portfolio main.tex
```

The assembled review PDF is written to
`build/A00049113_EEN1095_Full_Project_Portfolio.pdf`. Appendix A is provisional
until the final GenAI completeness and attribution audit is finished.

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
