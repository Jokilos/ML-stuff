#include <stdio.h>

// Function defined in x86-64
int x86_base(int x) {
    return x + 10;
}

extern int base_text(int x);
extern int call_text(int x);
extern int base_custom(int x);
extern int call_custom_text(int x);
extern int call_custom_myfuncs(int x);
extern int base_myfuncs(int x);
extern int call_myfuncs_custom(int x);

int real_main(void) {
    int r1 = base_text(10);          // 10 + 1
    int r2 = call_text(10);          // 10 + 1 + 2 + (10 + 10)
    int r3 = base_custom(20);        // 20 + 3
    int r4 = call_custom_text(20);   // 20 + 1 + 4
    int r5 = call_custom_myfuncs(20); // 20 + 6 + 5 + (20 + 10)
    int r6 = base_myfuncs(30);       // 30 + 6
    int r7 = call_myfuncs_custom(30); // 30 + 3 + 7

    printf("%d (11)\n", r1);   // 10 + 1
    printf("%d (33)\n", r2);   // 10 + 1 + 2 + 20
    printf("%d (23)\n", r3);   // 20 + 3
    printf("%d (25)\n", r4);   // 20 + 1 + 4
    printf("%d (61)\n", r5);   // 20 + 6 + 5 + 30
    printf("%d (36)\n", r6);   // 30 + 6
    printf("%d (40)\n", r7);   // 30 + 3 + 7

    return (r1 != 11) || (r2 != 33) || (r3 != 23) ||
           (r4 != 25) || (r5 != 61) || (r6 != 36) || (r7 != 40);
}
