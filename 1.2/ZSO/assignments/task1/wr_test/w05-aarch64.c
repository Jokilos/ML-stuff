int f(int x) {
    int z = 0;
    int res = 0;
    while (z < x) {
        z += 1;
        res += z;
    }
    
    return res;
}