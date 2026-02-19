#!/bin/bash

# ==============================================================================
# HPCC Value Extractor
#
# This script processes exactly one HPCC output file and extracts the raw values
# for the 7 core HPCC tests, outputting them in a YAML-compatible format.
#
# Usage: ./extract_hpcc_values.sh <hpccoutf_file>
# ==============================================================================

# Ensure exactly one file is provided
if [ "$#" -ne 1 ]; then
    echo "Error: This script requires exactly one input file."
    echo "Usage: $0 <hpccoutf_file>"
    exit 1
fi

# Check if the file exists
if [ ! -f "$1" ]; then
    echo "Error: File '$1' not found."
    exit 1
fi

AWK_SCRIPT='
BEGIN {
    FS="="
    # Mapping indices to YAML keys
    test_keys[1] = "HPL"; test_keys[2] = "DGEMM"; test_keys[3] = "STREAM"
    test_keys[4] = "PTRANS"; test_keys[5] = "FFT"; test_keys[6] = "RA"
    test_keys[7] = "MPIFFT"
}

# Helper to clean the line and return a numeric value
function extract_num(line) {
    val = line; sub(/[^=]*=/, "", val); return val + 0
}

# Helper to save the first valid value found
function save_val(idx, val) {
    if (!(idx in results)) {
        results[idx] = val
    }
}

# --- Extraction Patterns ---

# 1. HPL (Convert Tflops to Gflops)
/^HPL_Tflops=/ { save_val(1, extract_num($0) * 1000) }

# 2. DGEMM
/^MPIDG_Gflops=/ || /^StarDGEMM_Gflops=/ || /^DGEMM_Gflops=/ || \
/^SingleDGEMM_Gflops=/ || /^SingleDGEMM_NoF=/ || /^StarDGEMM_NoF=/ {
    save_val(2, extract_num($0))
}

# 3. STREAM
/^MPIS_Triad=/ || /^StarSTREAM_Triad=/ || /^SingleSTREAM_Triad=/ || /^STREAM_Triad=/ {
    save_val(3, extract_num($0))
}

# 4. PTRANS
/^PTRANS_GBs=/ { save_val(4, extract_num($0)) }

# 5. FFT
/^StarFFT_Gflops=/ || /^SingleFFT_Gflops=/ { save_val(5, extract_num($0)) }

# 6. RA
/^MPIRandomAccess_GUPs=/ { save_val(6, extract_num($0)) }

# 7. MPIFFT
/^MPIFFT_Gflops=/ { save_val(7, extract_num($0)) }

END {
    print "---"
    print "  output:"
    for (i=1; i<=7; i++) {
        key = test_keys[i]
        val = (i in results) ? sprintf("%.6e", results[i]) : "null"
        printf "    %s:\n", key
        printf "      value: %s\n", val
    }
}
'

awk "$AWK_SCRIPT" "$1"
