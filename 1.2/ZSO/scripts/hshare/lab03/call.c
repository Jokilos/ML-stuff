#include <unistd.h>
#include <sys/syscall.h>
#include <errno.h>

int main()
{
    int rc = syscall(SYS_chmod, "./f", 777); // oops
    if (rc == -1)
        return errno;
    return 0;
}
