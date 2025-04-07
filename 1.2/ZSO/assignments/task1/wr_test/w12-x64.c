#include <stdio.h>

extern int func_text1(int a, int b);
extern int func_text2(int a, int b);
extern int func_text3(int a, int b);
extern int func_custom1(int a, int b);
extern int func_custom2(int a, int b);
extern int func_myfuncs1(int a, int b);
extern int func_myfuncs2(int a, int b);
extern int func_myfuncs3(int a, int b);
extern int func_myfuncs4(int a, int b);

int real_main(void) {
    int r1 = func_text1(5, 3);
    int r2 = func_text2(5, 3);
    int r3 = func_text3(5, 3);
    int r4 = func_custom1(10, 20);
    int r5 = func_custom2(10, 20);
    int r6 = func_myfuncs1(15, 25);
    int r7 = func_myfuncs2(15, 25);
    int r8 = func_myfuncs3(15, 25);
    int r9 = func_myfuncs4(15, 25);

    printf("%d (8)\n", r1);    // 5 + 3 + 0
    printf("%d (11)\n", r2);   // 5 + 3 + 3
    printf("%d (14)\n", r3);   // 5 + 3 + 6
    printf("%d (31)\n", r4);   // 10 + 20 + 1
    printf("%d (34)\n", r5);   // 10 + 20 + 4
    printf("%d (42)\n", r6);   // 15 + 25 + 2
    printf("%d (45)\n", r7);   // 15 + 25 + 5
    printf("%d (47)\n", r8);   // 15 + 25 + 7
    printf("%d (48)\n", r9);   // 15 + 25 + 8

    return (r1 != 8) || (r2 != 11) || (r3 != 14) ||
           (r4 != 31) || (r5 != 34) ||
           (r6 != 42) || (r7 != 45) || (r8 != 47) || (r9 != 48);
}
