#!/bin/bash

cd "$(dirname "$0")"  # Ensure we're in the script's directory

SUCCESS_LOG="passed.txt"
FAIL_LOG="failed.txt"

# Clear logs
> "$SUCCESS_LOG"
> "$FAIL_LOG"

for dir in eval/problem*; do
    echo "Processing $dir"

    TEST_C="$dir/test.c"
    ASM_S="$dir/code.s"
    TEST_O="$dir/test.o"
    ASM_O="$dir/code.o"
    EXEC="$dir/test_exec"

    # Check required files exist
    if [[ ! -f "$TEST_C" || ! -f "$ASM_S" ]]; then
        echo "Missing files in $dir, skipping."
        continue
    fi

    # Compile C test file
    clang -c "$TEST_C" -target arm64-apple-macos -o "$TEST_O"
    if [[ $? -ne 0 ]]; then
        echo "Failed to compile test.c in $dir"
        continue
    fi

    # Assemble predicted assembly
    as -arch arm64 "$ASM_S" -o "$ASM_O"
    if [[ $? -ne 0 ]]; then
        echo "Failed to assemble code.s in $dir"
        echo "$dir" >> "$FAIL_LOG"
        continue
    fi

    # Link
    clang "$ASM_O" "$TEST_O" -o "$EXEC"
    if [[ $? -ne 0 ]]; then
        echo "Linking failed in $dir"
        echo "$dir" >> "$FAIL_LOG"
        continue
    fi

    # Run
    "$EXEC"
    if [[ $? -eq 0 ]]; then
        echo "Passed: $dir"
        echo "$dir" >> "$SUCCESS_LOG"
    else
        echo "Failed: $dir"
        echo "$dir" >> "$FAIL_LOG"
    fi
done

echo ""
echo "Passed: $(wc -l < $SUCCESS_LOG)"
echo "Failed: $(wc -l < $FAIL_LOG)"

TESTED=$(( $(wc -l < "$SUCCESS_LOG") + $(wc -l < "$FAIL_LOG") ))
echo "Tested: $TESTED"