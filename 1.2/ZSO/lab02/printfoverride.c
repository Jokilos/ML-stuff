#include <stdio.h>
#include <stdarg.h>

void my_printf(const char* fmt, ...)
{
    printf("[%p] ", fmt);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

int main()
{
   my_printf("printing %d + %d = %s\n", 2, 7, "no idea");
}