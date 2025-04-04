#include <stdio.h>

int outside(int a) {
    return a + 10;
}

int abc = 1;

int f(int);
int g(int);
int h(int);

int real_main(void) {
    // int x = f(10);
    // int y = g(10);
    // int z = h(10);
    // printf("%d", x + y + z);
    printf("%d %d %d\n", f(10), g(10), h(10));
    printf("%d %d %d\n", f(20), g(20), h(20));
    printf("%d %d %d\n", f(15), g(15), h(15));
    return 0;
}

