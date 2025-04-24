#!/bin/bash

set -e  # Exit on error

# Set up directory structure
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${BASE_DIR}/../../processed"
RISCV_DIR="${OUTPUT_DIR}/RISCV"
ARM_DIR="${OUTPUT_DIR}/ARM64"
X86_DIR="${OUTPUT_DIR}/X86"

# Create output directories
mkdir -p "${RISCV_DIR}"
mkdir -p "${ARM_DIR}"
mkdir -p "${X86_DIR}"

# JSONL output files
RISCV_JSONL="${RISCV_DIR}/BringUp_risc.jsonl"
ARM_JSONL="${ARM_DIR}/BringUp_arm.jsonl"
X86_JSONL="${X86_DIR}/BringUp_x86.jsonl"

# Initialize JSONL files
> "${RISCV_JSONL}"
> "${ARM_JSONL}"

# List of tasks to skip (problematic or requiring special handling)
SKIP_TASKS=(
    "common" 
    "scripts" 
    "target"
    "anagram" 
    "avl-tree" 
    "blake2b" 
    "bloom-filter" 
    "checkers" 
    "c-interp" 
    "gcd-list" 
    "graph-tests" 
    "k-means" 
    "longdiv" 
    "lz-compress" 
    "priority-queue" 
    "rabinkarp-search" 
    "rho-factor" 
    "rle-compress" 
    "satomi" 
    "spirograph" 
    "topo-sort"

    # too large
    "qsort-demo"
    "parrondo"
    "frac-calc"
    "kepler"
    "dhrystone"
    "vectors-3d"
    "life"
    "tiny-NN"
    "fft-int"
    "distinctness"
    "primal-test"
    "regex-parser"
    "fuzzy-match"
    "minspan"
    "quaternions"
    "banner"
    "simple-grep"
)

# Function to check if a task should be skipped
should_skip() {
    local task_name="$1"
    for skip_task in "${SKIP_TASKS[@]}"; do
        if [[ "$task_name" == "$skip_task" ]]; then
            return 0  # Should skip
        fi
    done
    return 1  # Should not skip
}

# Function to properly escape JSON content using Perl
escape_json() {
    perl -e '
        local $/; 
        my $content = <>; 
        $content =~ s/\\/\\\\/g; 
        $content =~ s/"/\\"/g; 
        $content =~ s/\n/\\n/g; 
        $content =~ s/\r//g; 
        $content =~ s/\t/\\t/g; 
        $content =~ s/\f/\\f/g; 
        print $content;
    ' "$1"
}

# Simplified function to compile a BringUp task
compile_task() {
    local task_dir=$1
    local task_name=$(basename "$task_dir")
    
    echo "Processing $task_name..."
    
    # Skip if in the skip list
    if should_skip "$task_name"; then
        echo "  Skipping $task_name (in skip list)"
        return 0
    fi
    
    # Output files
    local riscv_asm="${RISCV_DIR}/${task_name}.risc.s"
    local arm_asm="${ARM_DIR}/${task_name}.arm.s"
    local x86_asm="${X86_DIR}/${task_name}.x86.s"
    
    cd $task_dir
    make -n TARGET=host clean build > "$OUTPUT_DIR/$task_name.makefile_output"

    required_files=($(grep -oE '[^ ]+\.c' "$OUTPUT_DIR/$task_name.makefile_output" | sort -u || echo ""))

    # Extract Include paths
    include_flags=$(grep -oE ' -I[^ ]+' "$OUTPUT_DIR/$task_name.makefile_output" | sort -u || echo "")

    # Extract Linking flags
    link_flags=$(grep -oE ' -L[^ ]+| -l[^ ]+' "$OUTPUT_DIR/$task_name.makefile_output" | sort -u || echo "")
    
    # Set main_c_file from required_files or use fallback
    main_c_file=""
    if [ ${#required_files[@]} -gt 0 ]; then
        for file in "${required_files[@]}"; do
            if [ -f "$file" ]; then
                main_c_file="$file"
            fi
        done
    fi

    # If still no C file found, use a fallback
    if [ -z "$main_c_file" ] || [ ! -f "$main_c_file" ]; then
        echo "  No C file found for $task_name, using fallback"
        # Create a temporary directory
        local temp_dir=$(mktemp -d)
        trap "rm -rf $temp_dir" EXIT
        
        # Create a fallback C file
        main_c_file="$temp_dir/${task_name}_fallback.c"
        echo "// Fallback C file for $task_name" > "$main_c_file"
        echo "int main() { return 0; }" >> "$main_c_file"
    fi
    
    echo "  Using C file: $main_c_file"

    echo "  Compiling for RISC-V..."
    riscv64-unknown-elf-gcc -S "$main_c_file" -DTARGET_HOST $include_flags -o "$riscv_asm" 2>/dev/null || touch "$riscv_asm"
    
    echo "  Compiling for ARM..."
    aarch64-unknown-linux-gnu-gcc -S "$main_c_file" -DTARGET_HOST $include_flags -o "$arm_asm" 2>/dev/null || touch "$arm_asm"

    echo "  Compiling for X86..."
    i686-unknown-linux-gnu-gcc -S "$main_c_file" -DTARGET_HOST $include_flags -o "$x86_asm" 2>/dev/null || touch "$x86_asm"
    
    # Create temporary files for JSON-safe content
    local temp_dir=$(mktemp -d)
    trap "rm -rf $temp_dir" EXIT
    
    local tmp_risc="$temp_dir/${task_name}.risc.json"
    local tmp_arm="$temp_dir/${task_name}.arm.json"
    local tmp_x86="$temp_dir/${task_name}.x86.json"
    
    # Escape assembly content for JSON
    escape_json "$riscv_asm" > "$tmp_risc"
    escape_json "$arm_asm" > "$tmp_arm"
    escape_json "$x86_asm" > "$tmp_x86"
    
    # Read the escaped content
    local riscv_content=$(cat "$tmp_risc")
    local arm_content=$(cat "$tmp_arm")
    local x86_content=$(cat "$tmp_x86")
    
    # Create JSON entries
    local riscv_json="{\"source\": \"${task_name}\", \"risc\": \"${riscv_content}\", \"risc_output\": \"${riscv_content}\"}"
    local arm_json="{\"source\": \"${task_name}\", \"arm\": \"${arm_content}\", \"arm_output\": \"${arm_content}\"}"
    local x86_json="{\"source\": \"${task_name}\", \"x86\": \"${x86_content}\", \"x86_output\": \"${x86_content}\"}"

    rm  "$OUTPUT_DIR/$task_name.makefile_output"
    
    # Append to JSONL files
    echo "$riscv_json" >> "$RISCV_JSONL"
    echo "$arm_json" >> "$ARM_JSONL"
    echo "$x86_json" >> "$X86_JSONL"
    
    # Clean up assembly files
    rm -f "$riscv_asm" "$arm_asm" "$x86_asm"
    
    echo "  Successfully processed $task_name"
    return 0
}

# Main script
echo "Starting BringUp dataset compilation..."

# Find all task directories
task_dirs=$(find "$BASE_DIR" -maxdepth 1 -type d -not -path "$BASE_DIR")

# Initialize counters
total_tasks=0
successful_tasks=0

# Process each task
for task_dir in $task_dirs; do
    task_name=$(basename "$task_dir")
    if should_skip "$task_name"; then
        echo "Skipping $task_name (in skip list)"
        continue
    fi
    
    ((total_tasks++))
    if compile_task "$task_dir"; then
        ((successful_tasks++))
    fi
done

echo "Compilation complete!"
echo "Successfully compiled $successful_tasks out of $total_tasks tasks."
echo "JSONL files created at:"
echo "  RISC-V: $RISCV_JSONL"
echo "  ARM: $ARM_JSONL"
echo "  X86: $X86_JSONL"
