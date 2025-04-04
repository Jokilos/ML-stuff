#include <stdio.h>

extern int even(int n);
extern int odd(int n);

int real_main(void) {
    int result1 = even(4);  // Should return 1 (4 is even)
    int result2 = odd(5);   // Should return 1 (5 is odd)
    printf("even(4) = %d\n", result1);
    printf("odd(5) = %d\n", result2);
    return 0;
}
