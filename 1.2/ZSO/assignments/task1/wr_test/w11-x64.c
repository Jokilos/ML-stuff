#include <stdio.h>

extern int sum_3d_array(void);

int real_main(void) {
    unsigned int result = sum_3d_array();
    printf("Sum = %u\n", result);
    return 0;
}