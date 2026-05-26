# sparam-surrogate

Machine learning surrogate models for predicting PCB interconnect S-parameters and signal integrity metrics.

## Setup

```bash
conda env create -f environment.yml
conda activate meng
pip install -e .
```

## PDF reports

Build the executed notebook reports with:

```bash
make webpdf
```

The first WebPDF build may download Playwright's Chromium runtime into its
user cache for `nbconvert`; later builds reuse it. In an offline environment,
install it in advance with `playwright install chromium`.
