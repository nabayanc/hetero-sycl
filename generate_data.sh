#!/usr/bin/env bash
set -euo pipefail # Exit on error, unset variable, and error on pipe failure

# ─── CONFIGURATION ─────────────────────────────────────────────────────────
BUILD_DIR="build" 
CREATECSR_EXE="${BUILD_DIR}/createcsr"
DATASETS_DIR="DATASETS" 

# Define ranges for matrix generation from your set -x output
SIZES=(10000 25000 50000 75000 100000)
DENSITIES=(0.0001 0.0005 0.001 0.05 0.01)

# ─── PREPARATION ───────────────────────────────────────────────────────────

# Check if createcsr executable exists
if [[ ! -x "${CREATECSR_EXE}" ]]; then
  echo >&2 "Error: ${CREATECSR_EXE} not found or not executable."
  echo >&2 "Please build the project first (e.g., using make or ninja in the '${BUILD_DIR}' directory)."
  exit 1
fi

# Create the datasets directory if it doesn't exist
mkdir -p "${DATASETS_DIR}"
echo "==> Datasets will be stored in: ${DATASETS_DIR}"
echo "==> CreateCSR executable: ${CREATECSR_EXE}"
echo "---------------------------------------------------------------------"

# ─── MATRIX GENERATION AND VERIFICATION ───────────────────────────────────

for SIZE in "${SIZES[@]}"; do
  for DENSITY in "${DENSITIES[@]}"; do
    # Convert density to a string format suitable for filenames
    DSTR=$(printf "%s" "$DENSITY" | sed 's/\./_/')
    
    MTX_FILENAME="csr_r${SIZE}_c${SIZE}_d${DSTR}.mtx"
    MTX_FILEPATH="${DATASETS_DIR}/${MTX_FILENAME}"

    echo "==> Generating Matrix: ${MTX_FILENAME}"
    echo "    Size: ${SIZE}x${SIZE}, Target Density: ${DENSITY}" # Using 4 spaces for consistency
    echo "    Output Path: ${MTX_FILEPATH}"

    # Generate the matrix and capture its stdout/stderr
    # If createcsr fails (non-zero exit), set -e will stop the script here.
    generation_output=$("${CREATECSR_EXE}" \
      --rows    "${SIZE}" \
      --cols    "${SIZE}" \
      --density "${DENSITY}" \
      --output  "${MTX_FILEPATH}" 2>&1) 

    # Correctness Check 1: File existence and non-emptiness
    if [[ ! -f "${MTX_FILEPATH}" ]]; then
      echo "    ERROR: Matrix file ${MTX_FILEPATH} was NOT created."
      echo "    CreateCSR Output was:"
      echo "${generation_output}"
      echo "    -------------------------------------------------"
      continue # Skip to the next matrix
    fi
    if [[ ! -s "${MTX_FILEPATH}" ]]; then
      echo "    ERROR: Matrix file ${MTX_FILEPATH} is EMPTY."
      echo "    CreateCSR Output was:"
      echo "${generation_output}"
      echo "    -------------------------------------------------"
      continue 
    fi
    
    echo "    File ${MTX_FILEPATH} created successfully."
    # You can uncomment the line below if you want to see createcsr's output for every successful generation
    # echo "    CreateCSR Output: ${generation_output}"


    # Correctness Check 2: Read and validate MatrixMarket header
    # Use grep -m 1 to get the first non-comment line, avoids SIGPIPE with head
    header_line_output=$(grep -m 1 -v '^%' "${MTX_FILEPATH}")
    grep_exit_status=$? 

    if [[ "${grep_exit_status}" -ne 0 ]]; then
        # grep -m 1 -v exits 1 if no non-comment line is found, >1 for other errors.
        if [[ "${grep_exit_status}" -eq 1 ]]; then
            echo "    ERROR: No non-comment (header) line found in ${MTX_FILEPATH}."
        else
            echo "    ERROR: 'grep -m 1 -v' failed with exit status ${grep_exit_status} for ${MTX_FILEPATH}."
        fi
        echo "    -------------------------------------------------"
        continue # Skip to next matrix
    fi
    
    # Assign to header_line (primarily for clarity if needed later, though header_line_output can be used directly)
    header_line="${header_line_output}"

    if [[ -z "$header_line" ]]; then # Should be caught by grep_exit_status=1 already, but as a safeguard
        echo "    ERROR: Header line read from ${MTX_FILEPATH} is empty (this shouldn't happen if grep exited 0)."
        echo "    -------------------------------------------------"
        continue 
    fi

    # Parse the header line
    read -r file_rows file_cols file_nnz <<< "$header_line"

    # Validate parsed dimensions
    valid_header_info=true
    if [[ "$file_rows" -ne "$SIZE" ]]; then
        echo "    ERROR: Mismatch in ROWS. Expected ${SIZE}, got '${file_rows}' in ${MTX_FILEPATH}."
        valid_header_info=false
    fi
    if [[ "$file_cols" -ne "$SIZE" ]]; then
        echo "    ERROR: Mismatch in COLS. Expected ${SIZE}, got '${file_cols}' in ${MTX_FILEPATH}."
        valid_header_info=false
    fi
    
    # Check if NNZ is an integer and plausible (e.g., > 0 for positive density)
    # Regex to check if file_nnz is a non-negative integer
    if ! [[ "$file_nnz" =~ ^[0-9]+$ ]]; then
        echo "    ERROR: NNZ ('${file_nnz}') read from header in ${MTX_FILEPATH} is not a valid non-negative integer."
        valid_header_info=false
    elif [[ "$file_nnz" -le 0 ]] && (( $(echo "$DENSITY > 0" | bc -l) )); then # bc is fine if you confirmed it's installed
        # This warning is for cases where density is >0 but reported NNZ is 0.
        # For very small densities and sizes, NNZ could be 0 by chance, but less likely for your chosen SIZES.
        echo "    WARNING: NNZ is ${file_nnz} in ${MTX_FILEPATH} for a positive target density ${DENSITY}."
        # Depending on strictness, you might set valid_header_info=false here too.
    fi
    
    if $valid_header_info; then
        echo "    Header Check: OK (File reports Rows: ${file_rows}, Cols: ${file_cols}, NNZ: ${file_nnz})"
    else
        echo "    Header Check: FAILED. Please review errors for ${MTX_FILEPATH}."
    fi
    echo "    -------------------------------------------------"

  done # End density loop
done # End size loop

echo "====================================================================="
echo "All matrix generation tasks completed."
echo "Generated matrices are in: ${DATASETS_DIR}"
echo "====================================================================="