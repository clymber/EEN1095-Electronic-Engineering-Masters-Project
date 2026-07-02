## Build PDF reports from synced Jupytext/Jupyter notebooks.

NOTEBOOK_DIR := notebooks
REPORT_DIR := reports/pdf
TEMPLATE_DIR := nbconvert_templates
NOTEBOOK_STAMP_DIR := .notebook-stamps
THIS_MAKEFILE := $(lastword $(MAKEFILE_LIST))

PY_NOTEBOOKS := $(wildcard $(NOTEBOOK_DIR)/*.py)
IPYNB_NOTEBOOKS := $(wildcard $(NOTEBOOK_DIR)/*.ipynb)

PY_STEMS := $(basename $(notdir $(PY_NOTEBOOKS)))
IPYNB_STEMS := $(basename $(notdir $(IPYNB_NOTEBOOKS)))
NOTEBOOK_STEMS := $(sort $(PY_STEMS) $(IPYNB_STEMS))

PDF_TARGETS := $(addprefix $(REPORT_DIR)/,$(addsuffix .pdf,$(NOTEBOOK_STEMS)))
GENERATED_NOTEBOOKS := $(addprefix $(NOTEBOOK_DIR)/,$(addsuffix .ipynb,$(PY_STEMS)))
NOTEBOOK_RUN_STAMPS := $(addprefix $(NOTEBOOK_STAMP_DIR)/,$(addsuffix .executed,$(NOTEBOOK_STEMS)))

TEMPLATE_FILES := \
	$(TEMPLATE_DIR)/tight/conf.json \
	$(TEMPLATE_DIR)/tight/index.html.j2

WEBPDF_FLAGS ?= --allow-chromium-download
NOTEBOOK_EXECUTE_FLAGS ?= --ExecutePreprocessor.timeout=-1
NOTEBOOK_STREAM_FLAGS ?= --CoalesceStreamsPreprocessor.enabled=True

.PHONY: webpdf sync-notebooks run-notebooks clean distclean list-pdf-targets

webpdf: $(PDF_TARGETS)

sync-notebooks: $(addprefix $(NOTEBOOK_DIR)/,$(addsuffix .ipynb,$(PY_STEMS)))

run-notebooks: $(NOTEBOOK_RUN_STAMPS)

$(REPORT_DIR):
	mkdir -p $@

$(NOTEBOOK_STAMP_DIR):
	mkdir -p $@

$(NOTEBOOK_DIR)/%.ipynb: $(NOTEBOOK_DIR)/%.py
	source ./env_setup.sh && jupytext --sync $<

$(NOTEBOOK_STAMP_DIR)/%.executed: $(NOTEBOOK_DIR)/%.ipynb | $(NOTEBOOK_STAMP_DIR)
	SPARAM_SURROGATE_PROGRESS=final source ./env_setup.sh && jupyter nbconvert $< \
		--to notebook \
		--execute \
		--inplace \
		--ExecutePreprocessor.kernel_name="$${NOTEBOOK_KERNEL}" \
		$(NOTEBOOK_EXECUTE_FLAGS) \
		$(NOTEBOOK_STREAM_FLAGS)
	touch $@

$(REPORT_DIR)/%.pdf: $(NOTEBOOK_STAMP_DIR)/%.executed $(TEMPLATE_FILES) $(THIS_MAKEFILE) | $(REPORT_DIR)
	source ./env_setup.sh && jupyter nbconvert $(NOTEBOOK_DIR)/$*.ipynb \
		--to webpdf \
		$(WEBPDF_FLAGS) \
		$(NOTEBOOK_STREAM_FLAGS) \
		--ExecutePreprocessor.kernel_name="$${NOTEBOOK_KERNEL}" \
		--TagRemovePreprocessor.enabled=True \
		--TagRemovePreprocessor.remove_input_tags=remove-input \
		--template tight \
		--TemplateExporter.extra_template_basedirs=./$(TEMPLATE_DIR) \
		--output-dir $(REPORT_DIR) \
		--output $*

clean:
	rm -rf $(REPORT_DIR)/*

distclean: clean
	rm -rf $(NOTEBOOK_STAMP_DIR)
	rm -f $(GENERATED_NOTEBOOKS)

list-pdf-targets:
	@printf '%s\n' $(PDF_TARGETS)
