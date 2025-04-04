int f(int x){
    x += 5;
    int y = x + 1;
    if (y > 10) {
        return 100;
    }
    return -100;
}