int value1 = 100;
int value2 = 101;
int *value2_ptr = &value2;
int **value2_ptr_ptr = &value2_ptr;

int f() {
    *value2_ptr_ptr = &value1;
    return 0;
}
