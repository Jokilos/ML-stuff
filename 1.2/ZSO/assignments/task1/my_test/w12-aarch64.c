// Functions in default .text section (3 functions)
int func_text1(int a, int b) {
    return a + b;          // Base addition
}

int func_text2(int a, int b) {
    return a + b + 3;      // Add 3
}

int func_text3(int a, int b) {
    return a + b + 6;      // Add 6
}

// Functions in custom section .text.custom (2 functions)
__attribute__((section(".text.custom")))
int func_custom1(int a, int b) {
    return a + b + 1;      // Add 1
}

__attribute__((section(".text.custom")))
int func_custom2(int a, int b) {
    return a + b + 4;      // Add 4
}

// Functions in custom section .myfuncs (4 functions)
__attribute__((section(".myfuncs")))
int func_myfuncs1(int a, int b) {
    return a + b + 2;      // Add 2
}

__attribute__((section(".myfuncs")))
int func_myfuncs2(int a, int b) {
    return a + b + 5;      // Add 5
}

__attribute__((section(".myfuncs")))
int func_myfuncs3(int a, int b) {
    return a + b + 7;      // Add 7
}

__attribute__((section(".myfuncs")))
int func_myfuncs4(int a, int b) {
    return a + b + 8;      // Add 8
}
