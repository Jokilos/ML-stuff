extern int YOUR_DATA;
int bar(int*);


static int DATA = 42;

static void baz() {
    int a = 32 * 6;
}

int foo(int c) {
    DATA = YOUR_DATA;
    baz();
    return bar(&DATA);
}