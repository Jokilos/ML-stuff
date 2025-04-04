#include <stdio.h>

extern int global_var;
extern int (*func_ptr)(int, int);

int real_main(void) {
    printf("%d\n", global_var); // expected 42
    int result = func_ptr(5, 3);
    printf("%d\n", result); // expected 8
    return 0;
}
