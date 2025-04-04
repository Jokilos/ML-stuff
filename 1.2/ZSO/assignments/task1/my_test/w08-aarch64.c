int times_one(int x) {
    return x;
}

int times_two(int x) {
    return times_one(x) + x;
}

int times_three(int x) {
    return times_two(x) + x;
}

int times_four(int x) {
    return times_three(x) + x;
}

int times_five(int x) {
    return times_four(x) + x;
}

int times_six(int x) {
    return times_five(x) + x;
}

int (*mults[])(int) = {
    times_three,
    times_four,
    times_five,
    times_six
};
