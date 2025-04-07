#!/bin/bash

# Lista testów (np. test1, test2, test3)
tests=($(ls *-aarch64.c 2>/dev/null | sed 's/-aarch64\.c$//'))

mkdir -p outputs

# Liczniki
passed=0
failed=0

for name in "${tests[@]}"; do
    echo "Running test: $name"
    
    if ! make -B "ground-truth-test-$name"; then
        echo "Compilation of ground truth for test-$name ❌"
        ((failed++))
        continue
    fi

    # Kompilacja testu
    if ! make -B "test-$name"; then
        echo "Compilation failed for test-$name ❌"
        ((failed++))
        continue
    fi

    # Uruchomienie obu testów i zapisanie wyjścia do plików w katalogu "outputs"
    "./ground-truth-test-$name" > "outputs/output-ground-truth-$name.txt"
    "./test-$name" > "outputs/output-test-$name.txt"

    # Porównanie wyników
    if diff -q "outputs/output-ground-truth-$name.txt" "outputs/output-test-$name.txt" > /dev/null; then
        echo "Test $name passed ✅"
        ((passed++))
    else
        echo "Test $name failed ❌"
        ((failed++))
        diff "outputs/output-ground-truth-$name.txt" "outputs/output-test-$name.txt"
    fi
done

# Podsumowanie
echo "=============================="
echo "Total tests run: $((passed + failed))"
echo "Tests passed: $passed ✅"
echo "Tests failed: $failed ❌"
echo "=============================="

# Kod zakończenia 0 jeśli wszystkie testy przeszły, 1 jeśli są błędy
exit $((failed > 0 ? 1 : 0))
