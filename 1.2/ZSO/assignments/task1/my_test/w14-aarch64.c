// External function defined in x86-64 file
extern int x86_base(int x);

// Functions in .text (2 functions)
int base_text(int x) {
    return x + 1;
}

int call_text(int x) {
    return base_text(x) + 2 + x86_base(x);  // Calls .text and x86-64 function
}

// Functions in .text.custom (3 functions)
__attribute__((section(".text.custom")))
int base_custom(int x) {
    return x + 3;
}

__attribute__((section(".text.custom")))
int call_custom_text(int x) {
    return base_text(x) + 4;  // Calls .text function
}

int base_myfuncs(int x);

__attribute__((section(".text.custom")))
int call_custom_myfuncs(int x) {
    return base_myfuncs(x) + 5 + x86_base(x);  // Calls .myfuncs and x86-64 function
}

// Functions in .myfuncs (2 functions)
__attribute__((section(".myfuncs")))
int base_myfuncs(int x) {
    return x + 6;
}

__attribute__((section(".myfuncs")))
int call_myfuncs_custom(int x) {
    return base_custom(x) + 7;  // Calls .text.custom function
}
