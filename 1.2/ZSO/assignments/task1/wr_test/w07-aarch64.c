// Define a global variable
int global_var = 42;

int function_with_offset_0_probably(int x, int y) {
    return x + y + x + y + x + y;
}

// Define a function
int add_numbers(int a, int b) {
    return a + b;
}

// Define a function pointer initialized to point to add_numbers
typedef int (*func_ptr_t)(int, int);
func_ptr_t func_ptr = add_numbers;
