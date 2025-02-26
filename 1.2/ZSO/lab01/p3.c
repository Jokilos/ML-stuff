#include <stdio.h>

int f(int b) {
    return b + b;
}

char g(int a, int b, int c) {
    if (a > b)
        return 'a';
    if (b > c)
        return 'b';
    return 'c';
}

int main() {
    printf("f(42) is %d\n", f(42));
}