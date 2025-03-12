#include <unistd.h>
#include <string.h>

void write_message(const char *message, int length) {
    write(STDOUT_FILENO, message, length);
}

int main() {
    write_message("Hello, world!\n", 14);
    return 0;
}

