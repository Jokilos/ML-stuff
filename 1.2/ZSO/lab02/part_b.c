int foo(); // oopsie

int bar(int* arg) {
    return *arg + 4;
}

int YOUR_DATA = 1337;

int main() {
    return foo();
}