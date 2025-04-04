#include <stdio.h>

extern int* value2_ptr;

int f();

int real_main(void) {
    printf("%d\n", *value2_ptr); // expected 101
    f();
    printf("%d\n", *value2_ptr); // expected 102
    return 0;
}

