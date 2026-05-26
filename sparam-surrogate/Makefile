## Build PDF reports from synced Jupytext/Jupyter notebooks.

NOTEBOOK_DIR := notebooks
REPORT_DIR := reports/pdf
TEMPLATE_DIR := nbconvert_templates

PY_NOTEBOOKS := $(wildcard $(NOTEBOOK_DIR)/*.py)
IPYNB_NOTEBOOKS := $(wildcard $(NOTEBOOK_DIR)/*.ipynb)

PY_STEMS := $(basename $(notdir $(PY_NOTEBOOKS)))
IPYNB_STEMS := $(basename $(notdir $(IPYNB_NOTEBOOKS)))
NOTEBOOK_STEMS := $(sort $(PY_STEMS) $(IPYNB_STEMS))

PDF_TARGETS := $(addprefix $(REPORT_DIR)/,$(addsuffix .pdf,$(NOTEBOOK_STEMS)))

TEMPLATE_FILES := \
	$(TEMPLATE_DIR)/tight/conf.json \
	$(TEMPLATE_DIR)/tight/index.html.j2

EXECUTE ?= --execute
# WebPDF uses Playwright. Permit its browser runtime to be installed on first use.
WEBPDF_FLAGS ?= --allow-chromium-download

.PHONY: webpdf sync-notebooks clean-pdf list-pdf-targets

webpdf: $(PDF_TARGETS)

sync-notebooks: $(addprefix $(NOTEBOOK_DIR)/,$(addsuffix .ipynb,$(PY_STEMS)))

$(REPORT_DIR):
	mkdir -p $@

$(NOTEBOOK_DIR)/%.ipynb: $(NOTEBOOK_DIR)/%.py
	jupytext --sync $<

$(REPORT_DIR)/%.pdf: $(NOTEBOOK_DIR)/%.ipynb $(TEMPLATE_FILES) | $(REPORT_DIR)
	jupyter nbconvert $< \
		--to webpdf \
		$(WEBPDF_FLAGS) \
		$(EXECUTE) \
		--template tight \
		--TemplateExporter.extra_template_basedirs=./$(TEMPLATE_DIR) \
		--output-dir $(REPORT_DIR) \
		--output $*

clean-pdf:
	rm -rf $(REPORT_DIR)/*

list-pdf-targets:
	@printf '%s\n' $(PDF_TARGETS)
