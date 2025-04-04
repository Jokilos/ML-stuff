int outside(int);
extern int abc;

int f(int x){
    if (x > 10) {
        if (x < 20) {
            return 40;
        }
        return 50;
    }
    return 60;
}

int g(int x) {
    int y = 0;
    int c = 0;
    while (c < x) {
        y += outside(x + 5);
        y += f(x + 3);
        c += 1;
    }
    return y;
}

int h(int x){
    abc = x + abc;
    return g(abc);
}
