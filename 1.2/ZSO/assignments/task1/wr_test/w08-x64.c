#include <stdio.h>

extern int (*mults[])(int);

int real_main(void) {
    printf("%d\n", mults[0](10));
    printf("%d\n", mults[1](10));
    printf("%d\n", mults[2](10));
    printf("%d\n", mults[3](10));
    return 0;
}

