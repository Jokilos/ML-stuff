#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>   /* mmap(), mprotect() */

void writer(int signum) {
    write(1, "12", 2);
}
