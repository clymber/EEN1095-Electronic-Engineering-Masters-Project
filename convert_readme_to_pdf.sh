#!/usr/bin/env bash

# Convert a Markdown README to a readable, linked A4 PDF.
#
# Usage:
#   ./convert_readme_to_pdf.sh [INPUT.md [OUTPUT.pdf]]
#
# With no arguments, the script converts README.md beside this script to
# README.pdf. With only an input path, it writes a same-stem PDF beside the
# input file.

set -euo pipefail

script_directory=$(
  CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd
)

show_usage() {
  sed -n '3,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

fail() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_usage
  exit 0
fi

if (( $# > 2 )); then
  show_usage >&2
  fail "Expected at most an input path and an output path."
fi

input_argument=${1:-"$script_directory/README.md"}
if (( $# == 2 )); then
  output_argument=$2
elif (( $# == 1 )); then
  if [[ "$input_argument" == *.* ]]; then
    output_argument="${input_argument%.*}.pdf"
  else
    output_argument="${input_argument}.pdf"
  fi
else
  output_argument="$script_directory/README.pdf"
fi

[[ -f "$input_argument" ]] || fail "Input file not found: $input_argument"

input_directory=$(
  CDPATH= cd -- "$(dirname -- "$input_argument")" && pwd
)
input_path="$input_directory/$(basename -- "$input_argument")"

output_parent=$(dirname -- "$output_argument")
[[ -d "$output_parent" ]] || fail "Output directory not found: $output_parent"
output_directory=$(CDPATH= cd -- "$output_parent" && pwd)
output_name=$(basename -- "$output_argument")
output_path="$output_directory/$output_name"

[[ "$input_path" != "$output_path" ]] || \
  fail "Input and output paths must be different."

declare -a pandoc_command
if [[ -n "${PANDOC_BIN:-}" ]]; then
  [[ -x "$PANDOC_BIN" ]] || fail "PANDOC_BIN is not executable: $PANDOC_BIN"
  pandoc_command=("$PANDOC_BIN")
elif pandoc_path=$(command -v pandoc 2>/dev/null); then
  pandoc_command=("$pandoc_path")
elif conda_path=$(command -v conda 2>/dev/null); then
  if "$conda_path" run -n meng pandoc --version >/dev/null 2>&1; then
    pandoc_command=("$conda_path" run -n meng pandoc)
  else
    fail "Pandoc was not found on PATH or in the 'meng' Conda environment."
  fi
else
  fail "Pandoc was not found on PATH."
fi

if [[ -n "${XELATEX_BIN:-}" ]]; then
  [[ -x "$XELATEX_BIN" ]] || \
    fail "XELATEX_BIN is not executable: $XELATEX_BIN"
  xelatex_path=$XELATEX_BIN
elif xelatex_path=$(command -v xelatex 2>/dev/null); then
  :
elif [[ -x /Library/TeX/texbin/xelatex ]]; then
  xelatex_path=/Library/TeX/texbin/xelatex
else
  fail "XeLaTeX was not found on PATH."
fi

temporary_directory=${TMPDIR:-/tmp}
lua_filter=$(mktemp "${temporary_directory%/}/readme-pdf-filter.XXXXXX")
trap 'rm -f -- "$lua_filter"' EXIT

cat > "$lua_filter" <<'LUA'
local table_widths = {
  Evaluation = {0.20, 0.46, 0.34},
  ["No."] = {0.06, 0.36, 0.58},
  Label = {0.08, 0.38, 0.54},
}

function Table(tbl)
  local widths = nil
  if #tbl.head.rows > 0 and #tbl.head.rows[1].cells > 0 then
    local first_cell = tbl.head.rows[1].cells[1]
    widths = table_widths[pandoc.utils.stringify(first_cell)]
  end

  if not widths then
    widths = {}
    for index = 1, #tbl.colspecs do
      widths[index] = 1 / #tbl.colspecs
    end
  end

  if #widths == #tbl.colspecs then
    for index, width in ipairs(widths) do
      tbl.colspecs[index][2] = width
    end
  end
  return tbl
end

function Code(code)
  if FORMAT:match("latex") then
    return pandoc.RawInline("latex", "\\nolinkurl{" .. code.text .. "}")
  end
end
LUA

"${pandoc_command[@]}" "$input_path" \
  --from=gfm \
  --standalone \
  --lua-filter="$lua_filter" \
  --pdf-engine="$xelatex_path" \
  --resource-path="$input_directory" \
  --toc \
  --toc-depth=2 \
  -V geometry:margin=0.7in \
  -V fontsize=10pt \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolor=blue \
  -V papersize=a4 \
  -o "$output_path"

[[ -s "$output_path" ]] || fail "Pandoc did not create a non-empty PDF."
printf 'Created %s\n' "$output_path"
