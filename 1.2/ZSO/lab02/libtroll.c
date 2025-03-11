#define _GNU_SOURCE
#include <stdio.h>
#include <stdarg.h>
#include <dlfcn.h>

// Function pointer for the real printf
static int (*real_printf)(const char *, ...) = NULL;

// Our hijacked printf function
int printf(const char *fmt, ...)
{
    if (!real_printf) {
        // Resolve the real printf function
        real_printf = dlsym(RTLD_NEXT, "printf");
        if (!real_printf) {
            return -1;
        }
    }

    // Print the format string address
    real_printf("[Format String Address: %p] ", fmt);

    // Forward the original printf call
    va_list args;
    va_start(args, fmt);
    int result = real_printf(fmt, args);
    va_end(args);

    return result;
}