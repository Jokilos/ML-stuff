#include <stdio.h>

extern int x;

int f(){
    printf("Hello there!\n");
    x += 10;
    return x;
    // return 0;
}
