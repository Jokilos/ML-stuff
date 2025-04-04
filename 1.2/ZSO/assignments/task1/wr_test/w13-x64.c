#include <stdio.h>

// Global variable defined in x86-64
int global_data_x86 = 50;

extern int get_data_text1(int x);
extern int get_data_text2(int x);
extern int get_data_custom1(int x);
extern int get_data_custom2(int x);
extern int get_data_custom3(int x);
extern int get_data_myfuncs1(int x);

int real_main(void) {
    int r1 = get_data_text1(10);      // 100 + 10 + 50
    int r2 = get_data_text2(10);      // 200 + 10 + 5
    int r3 = get_data_custom1(20);    // 100 + 20 + 1
    int r4 = get_data_custom2(20);    // 200 + 20 + 2 + 50
    int r5 = get_data_custom3(20);    // 300 + 20 + 3
    int r6 = get_data_myfuncs1(30);   // 300 + 30 + 4 + 50

    printf("%d (160)\n", r1);  // 100 + 10 + 50
    printf("%d (215)\n", r2);  // 200 + 10 + 5
    printf("%d (121)\n", r3);  // 100 + 20 + 1
    printf("%d (272)\n", r4);  // 200 + 20 + 2 + 50
    printf("%d (323)\n", r5);  // 300 + 20 + 3
    printf("%d (384)\n", r6);  // 300 + 30 + 4 + 50

    return 0;
}
