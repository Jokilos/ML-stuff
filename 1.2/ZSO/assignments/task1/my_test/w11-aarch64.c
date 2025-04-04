#include <stdio.h>

unsigned int arr[2][3][4] = {
    {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
    {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}
};

// Funkcja mnożenia za pomocą dodawania
unsigned int multiply(unsigned int a, unsigned int b) {
    unsigned int result = 0;
    unsigned int count = 0;
    while (count != b) {
        result += a;
        count += 1;
    }
    return result;
}

unsigned int sum_3d_array(void) {
    unsigned int sum = 0;
    unsigned int* base_ptr = &arr[0][0][0]; // Wskaźnik na początek tablicy (adres w bajtach)
    unsigned int i = 0;
    unsigned int j = 0;
    unsigned int k = 0;

    while (i != 2) {
        j = 0;
        while (j != 3) {
            k = 0;
            while (k != 4) {
                // Oblicz przesunięcie w elementach: 12*i + 4*j + k
                unsigned int offset_elements = 0;
                offset_elements += multiply(12, i); // 12 * i
                offset_elements += multiply(4, j);  // 4 * j
                offset_elements += k;              // + k

                // Oblicz przesunięcie w bajtach: offset_elements * 4
                unsigned int offset_bytes = multiply(offset_elements, 4);

                // Oblicz adres docelowy
                unsigned int* current_ptr = (unsigned int*)((char*)base_ptr + offset_bytes);
                sum += *current_ptr;
                printf("arr[%u][%u][%u] = %u\n", i, j, k, *current_ptr);
                k += 1;
            }
            j += 1;
        }
        i += 1;
    }
    return sum;
}
