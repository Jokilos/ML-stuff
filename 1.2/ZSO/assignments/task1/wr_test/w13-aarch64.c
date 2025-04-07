// Global variables defined in AArch64
int global_data1 = 100;
int global_data2 = 200;
int global_data3 = 300;

// External global variable (defined in x86-64 file)
extern int global_data_x86;

// Functions in .text (2 functions)
int get_data_text1(int x) {
    return global_data1 + x + global_data_x86;
}

int get_data_text2(int x) {
    return global_data2 + x + 5;
}

// Functions in .text.custom (3 functions)
__attribute__((section(".text.custom")))
int get_data_custom1(int x) {
    return global_data1 + x + 1;
}

__attribute__((section(".text.custom")))
int get_data_custom2(int x) {
    return global_data2 + x + 2 + global_data_x86;
}

__attribute__((section(".text.custom")))
int get_data_custom3(int x) {
    return global_data3 + x + 3;
}

// Functions in .myfuncs (1 function)
__attribute__((section(".myfuncs")))
int get_data_myfuncs1(int x) {
    return global_data3 + x + 4 + global_data_x86;
}

