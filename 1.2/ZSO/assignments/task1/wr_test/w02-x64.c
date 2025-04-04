#include <stdio.h>

int x;

int f();

int real_main(void) {
    x = 0;

    printf("%d\n", f());
    return 0;
}

