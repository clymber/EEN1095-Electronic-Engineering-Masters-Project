#!/usr/bin/env bash

# Build the assessor-facing repository archive from the current working tree.
# Generated evidence is intentionally Git-ignored, so git archive is not suitable.

set -Eeuo pipefail

readonly ARCHIVE_NAME="A00049113_project_repository.zip"
readonly MAX_ARCHIVE_BYTES=50000000
readonly EXPECTED_SELECTED_MODELS=8

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

require_command() {
    local command_name=$1

    command -v "$command_name" >/dev/null 2>&1 \
        || fail "required command not found: $command_name"
}

if (( $# != 0 )); then
    fail "usage: $(basename "$0")"
fi

for command_name in \
    awk basename cp dirname find git jq mkdir mktemp mv rm sed sort unzip wc zip;
do
    require_command "$command_name"
done

script_dir=$(CDPATH= cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null) \
    || fail "the script must be inside a Git working tree"

[[ "$script_dir" == "$repo_root" ]] \
    || fail "place $(basename "$0") in the repository root"

readonly script_dir
readonly repo_root
readonly output_path="$repo_root/$ARCHIVE_NAME"
readonly readme_converter="$repo_root/convert_readme_to_pdf.sh"
readonly selected_file="sparam-surrogate/outputs/models/selected.json"

[[ ! -d "$output_path" && ! -L "$output_path" ]] \
    || fail "archive destination must not be a directory or symbolic link"
[[ -x "$readme_converter" ]] \
    || fail "README converter is missing or not executable"

printf 'Refreshing README.pdf...\n'
"$readme_converter" "$repo_root/README.md" "$repo_root/README.pdf" \
    || fail "could not convert README.md to PDF"

temp_parent=${TMPDIR:-/tmp}
temp_parent=${temp_parent%/}
work_dir=$(mktemp -d "$temp_parent/pack_portfolio.XXXXXX") \
    || fail "could not create a temporary directory"
publish_dir=""

cleanup() {
    local exit_status=$?

    trap - EXIT
    if [[ -n "$work_dir" && -d "$work_dir" \
        && "$work_dir" == */pack_portfolio.* ]]; then
        rm -rf "$work_dir"
    fi
    if [[ -n "$publish_dir" && -d "$publish_dir" \
        && "$publish_dir" == "$repo_root"/.pack_portfolio_publish.* ]]; then
        rm -rf "$publish_dir"
    fi
    exit "$exit_status"
}
trap cleanup EXIT

# Keep the candidate next to the destination. The final rename is then atomic even
# when TMPDIR is mounted on a different filesystem from the repository.
publish_dir=$(mktemp -d "$repo_root/.pack_portfolio_publish.XXXXXX") \
    || fail "could not create a temporary publish directory"

readonly work_dir
readonly publish_dir
readonly staging_dir="$work_dir/staging"
readonly verification_dir="$work_dir/verification"
readonly candidate_archive="$publish_dir/$ARCHIVE_NAME"
readonly manifest_file="$work_dir/archive-files.txt"
readonly tracked_manifest="$work_dir/tracked-files.bin"

mkdir -p "$staging_dir" "$verification_dir"

copy_file() {
    local relative_path=$1
    local source_path="$repo_root/$relative_path"
    local destination_path="$staging_dir/$relative_path"

    [[ "$relative_path" != *$'\n'* && "$relative_path" != *$'\r'* ]] \
        || fail "newline characters are not supported in paths"
    [[ "$relative_path" != /* \
        && "$relative_path" != ".." \
        && "$relative_path" != ../* \
        && "$relative_path" != */../* \
        && "$relative_path" != */.. ]] \
        || fail "unsafe repository path: $relative_path"
    [[ ! -L "$source_path" ]] \
        || fail "symbolic links are not supported: $relative_path"
    [[ -f "$source_path" ]] \
        || fail "required working-tree file is missing: $relative_path"

    mkdir -p "$(dirname "$destination_path")"
    cp -p "$source_path" "$destination_path"
}

copy_selected_run() {
    local run_path=$1
    local excluded_artifact=$2
    local source_root="$repo_root/sparam-surrogate/$run_path"
    local source_path
    local relative_path
    local run_relative_path
    local run_manifest

    [[ -d "$source_root" ]] \
        || fail "selected run directory is missing: sparam-surrogate/$run_path"

    run_manifest=$(mktemp "$work_dir/selected-run-files.XXXXXX") \
        || fail "could not create a selected-run manifest"
    find "$source_root" -type f -print0 > "$run_manifest" \
        || fail "could not enumerate selected run: $run_path"

    while IFS= read -r -d '' source_path; do
        relative_path=${source_path#"$repo_root/"}
        run_relative_path=${source_path#"$source_root/"}

        if [[ -n "$excluded_artifact" \
            && "$relative_path" == "sparam-surrogate/$excluded_artifact" ]]; then
            continue
        fi

        case "$run_relative_path" in
            config_resolved.json|environment.json|manifest.json|metadata.json|\
                metrics.json|training_history.csv|validation_results.csv|\
                preprocessors.joblib|model.joblib|model.keras|figures/*.png|\
                figures/*.pdf|figures/*.svg)
                ;;
            *)
                continue
                ;;
        esac

        copy_file "$relative_path"
    done < "$run_manifest"
}

printf 'Packaging the current working tree...\n'

# Start with tracked working-tree files. Large/source datasets, generated outputs,
# build intermediates, and the reference PDF library are added selectively below.
tracked_file_count=0
excluded_tracked_count=0
git -C "$repo_root" ls-files -z > "$tracked_manifest" \
    || fail "could not enumerate tracked working-tree files"
while IFS= read -r -d '' relative_path; do
    case "$relative_path" in
        Datasets/Readme.md)
            ;;
        "$ARCHIVE_NAME"|Datasets/*|References/*.pdf|References/*.PDF|\
            project_portfolio/build/*|sparam-surrogate/data/*|\
            sparam-surrogate/outputs/*|sparam-surrogate/reports/pdf/*)
            excluded_tracked_count=$((excluded_tracked_count + 1))
            continue
            ;;
    esac

    copy_file "$relative_path"
    tracked_file_count=$((tracked_file_count + 1))
done < "$tracked_manifest"

# Include the packaging scripts even before they have been added to Git.
copy_file "pack_portfolio.sh"
copy_file "convert_readme_to_pdf.sh"

generated_files=(
    "README.pdf"
    "project_portfolio/build/A00049113_EEN1095_Full_Project_Portfolio.pdf"
    "project_portfolio/build/A00049113_EEN1095_Research_Paper.pdf"
    "project_portfolio/evidence/page4_response_evidence.pdf"
    "project_portfolio/evidence/page4_response_evidence.png"
    "sparam-surrogate/notebooks/nb01_dataset_exploration.ipynb"
    "sparam-surrogate/notebooks/nb02_data_preprocessing.ipynb"
    "sparam-surrogate/notebooks/nb03_non_neural_modelling.ipynb"
    "sparam-surrogate/notebooks/nb04_neural_baseline.ipynb"
    "sparam-surrogate/notebooks/nb05_curve_neural_model.ipynb"
    "sparam-surrogate/notebooks/nb06_full_smatrix_physics.ipynb"
    "sparam-surrogate/notebooks/nb07_selected_models_evaluation_analysis.ipynb"
    "sparam-surrogate/notebooks/nb08_appendix_d_model_graphs.ipynb"
    "sparam-surrogate/reports/pdf/nb01_dataset_exploration.pdf"
    "sparam-surrogate/reports/pdf/nb02_data_preprocessing.pdf"
    "sparam-surrogate/reports/pdf/nb03_non_neural_modelling.pdf"
    "sparam-surrogate/reports/pdf/nb04_neural_baseline.pdf"
    "sparam-surrogate/reports/pdf/nb05_curve_neural_model.pdf"
    "sparam-surrogate/reports/pdf/nb06_full_smatrix_physics.pdf"
    "sparam-surrogate/reports/pdf/nb07_selected_models_evaluation_analysis.pdf"
)

for relative_path in "${generated_files[@]}"; do
    copy_file "$relative_path"
done

copy_file "$selected_file"

selected_models_tsv=$(jq -er '
    .models
    | to_entries
    | sort_by(.key)[]
    | [.key, .value.run_path, .value.artifact_path]
    | @tsv
' "$repo_root/$selected_file") \
    || fail "could not read selected model paths from $selected_file"

selected_model_names=()
selected_run_paths=()
selected_artifact_paths=()
random_forest_artifact=""

while IFS=$'\t' read -r model_name run_path artifact_path; do
    [[ -n "$model_name" && -n "$run_path" && -n "$artifact_path" ]] \
        || fail "incomplete selected model entry in $selected_file"
    [[ "$run_path" =~ ^outputs/runs/[[:alnum:]][[:alnum:]_.-]*$ ]] \
        || fail "unsafe selected run path: $run_path"
    [[ "$artifact_path" == "$run_path/"* \
        && "$artifact_path" != *"/../"* \
        && "$artifact_path" != */.. ]] \
        || fail "unsafe selected artifact path: $artifact_path"

    selected_model_names[${#selected_model_names[@]}]="$model_name"
    selected_run_paths[${#selected_run_paths[@]}]="$run_path"
    selected_artifact_paths[${#selected_artifact_paths[@]}]="$artifact_path"

    if [[ "$model_name" == "random_forest" ]]; then
        random_forest_artifact=$artifact_path
        copy_selected_run "$run_path" "$artifact_path"
    else
        [[ -f "$repo_root/sparam-surrogate/$artifact_path" ]] \
            || fail "selected model artifact is missing: $artifact_path"
        copy_selected_run "$run_path" ""
    fi
done <<< "$selected_models_tsv"

(( ${#selected_model_names[@]} == EXPECTED_SELECTED_MODELS )) \
    || fail "expected $EXPECTED_SELECTED_MODELS selected models, found \
${#selected_model_names[@]}"
[[ -n "$random_forest_artifact" ]] \
    || fail "the selected Random Forest artifact was not identified"

# Feed zip a sorted manifest so archive entry order is stable and paths with spaces
# remain intact. -X strips platform-specific extra fields; -9 maximises compression.
(
    cd "$staging_dir"
    find . -type f -print | LC_ALL=C sort | sed 's#^\./##' > "$manifest_file"
)
[[ -s "$manifest_file" ]] || fail "the archive manifest is empty"

(
    cd "$staging_dir"
    zip -q -9 -X "$candidate_archive" -@ < "$manifest_file"
)

archive_bytes=$(wc -c < "$candidate_archive")
archive_bytes=${archive_bytes//[[:space:]]/}
[[ "$archive_bytes" =~ ^[0-9]+$ ]] \
    || fail "could not determine the archive size"

if (( archive_bytes > MAX_ARCHIVE_BYTES )); then
    archive_mb=$(awk -v bytes="$archive_bytes" \
        'BEGIN { printf "%.2f", bytes / 1000000 }')
    fail "candidate archive is ${archive_mb} MB; limit is 50.00 MB"
fi

unzip -tq "$candidate_archive" >/dev/null \
    || fail "ZIP integrity check failed"
unzip -q "$candidate_archive" -d "$verification_dir" \
    || fail "could not extract the candidate ZIP for verification"

verify_file() {
    local relative_path=$1

    [[ -f "$verification_dir/$relative_path" ]] \
        || fail "archive verification failed; missing: $relative_path"
}

verify_file "README.md"
verify_file "pack_portfolio.sh"
verify_file "convert_readme_to_pdf.sh"
verify_file "project_portfolio/evidence/README.md"
verify_file "sparam-surrogate/environment.yml"
verify_file "sparam-surrogate/pyproject.toml"
verify_file "sparam-surrogate/configs/default.json"
verify_file "sparam-surrogate/outputs/models/selected.json"

evidence_files=(
    "project_portfolio/evidence/appendix_d_neural_model_inventory.csv"
    "project_portfolio/evidence/full_matrix_physics_diagnostics.csv"
    "project_portfolio/evidence/metric_reproduction_checks.csv"
    "project_portfolio/evidence/metric_reproduction_summary.csv"
    "project_portfolio/evidence/page4_common_target.csv"
    "project_portfolio/evidence/page4_full_matrix_comparison.csv"
    "project_portfolio/evidence/paired_bootstrap_transitions.csv"
    "project_portfolio/evidence/response_design_5491.csv"
    "project_portfolio/evidence/selected_run_provenance.csv"
)

for relative_path in "${evidence_files[@]}"; do
    verify_file "$relative_path"
done

for relative_path in "${generated_files[@]}"; do
    verify_file "$relative_path"
done

for relative_path in \
    "sparam-surrogate/notebooks/nb01_dataset_exploration.py" \
    "sparam-surrogate/notebooks/nb02_data_preprocessing.py" \
    "sparam-surrogate/notebooks/nb03_non_neural_modelling.py" \
    "sparam-surrogate/notebooks/nb04_neural_baseline.py" \
    "sparam-surrogate/notebooks/nb05_curve_neural_model.py" \
    "sparam-surrogate/notebooks/nb06_full_smatrix_physics.py" \
    "sparam-surrogate/notebooks/nb07_selected_models_evaluation_analysis.py" \
    "sparam-surrogate/notebooks/nb08_appendix_d_model_graphs.py"; do
    verify_file "$relative_path"
done

[[ -d "$verification_dir/project_portfolio" ]] \
    || fail "archive verification failed; project_portfolio/ is missing"
[[ -d "$verification_dir/sparam-surrogate/src/sparam_surrogate" ]] \
    || fail "archive verification failed; source package is missing"
[[ -d "$verification_dir/sparam-surrogate/tests" ]] \
    || fail "archive verification failed; tests are missing"

for index in "${!selected_run_paths[@]}"; do
    run_path=${selected_run_paths[$index]}
    artifact_path=${selected_artifact_paths[$index]}
    model_name=${selected_model_names[$index]}

    for record_name in \
        config_resolved.json environment.json manifest.json metadata.json metrics.json;
    do
        verify_file "sparam-surrogate/$run_path/$record_name"
    done

    run_directory="$verification_dir/sparam-surrogate/$run_path"
    if [[ ! -f "$run_directory/training_history.csv" \
        && ! -f "$run_directory/validation_results.csv" ]];
    then
        fail "archive verification failed; no history/results for $run_path"
    fi

    if [[ "$model_name" == "random_forest" ]]; then
        [[ ! -e "$verification_dir/sparam-surrogate/$artifact_path" ]] \
            || fail "oversized Random Forest artifact entered the archive"
    else
        verify_file "sparam-surrogate/$artifact_path"
    fi
done

[[ ! -e "$verification_dir/.git" ]] \
    || fail "Git metadata entered the archive"
[[ ! -e "$verification_dir/sparam-surrogate/data" ]] \
    || fail "generated/raw data entered the archive"

if [[ -d "$verification_dir/References" \
    && -n "$(find "$verification_dir/References" -type f \
        \( -name '*.pdf' -o -name '*.PDF' \) -print -quit)" ]]; then
    fail "reference-library PDFs entered the archive"
fi

[[ ! -d "$output_path" && ! -L "$output_path" ]] \
    || fail "archive destination became a directory or symbolic link"
mv -f "$candidate_archive" "$output_path"

archive_mb=$(awk -v bytes="$archive_bytes" \
    'BEGIN { printf "%.2f", bytes / 1000000 }')
printf 'Created %s\n' "$output_path"
printf 'Size: %s bytes (%s MB; limit 50.00 MB)\n' \
    "$archive_bytes" "$archive_mb"
printf 'Included %d tracked files and %d selected model records.\n' \
    "$tracked_file_count" "${#selected_model_names[@]}"
printf 'Excluded %d tracked generated/oversized files and the Random Forest binary.\n' \
    "$excluded_tracked_count"
