#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <sys/mman.h>   /* mmap(), mprotect() */

__attribute__((used)) ssize_t (*write_ptr)(int, const void *, size_t) = write;

static uint8_t my_code[] = {
/*0:*/  0x55, //push//rbp
/*1:*/  0x48, 0x89, 0xe5, //mov//rbp,rsp
/*4:*/  0x48, 0x83, 0xec, 0x10, //sub//rsp,0x10
/*8:*/  0x89, 0x7d, 0xfc, //mov//DWORD//PTR//[rbp-0x4],edi
/*b:*/  0xba, 0x02, 0x00, 0x00, 0x00, //mov//edx,0x2
/*10:*/  0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, //movabs//rax,0x0
/*17:*/  0x00, 0x00, 0x00, 
/*12:*/  //R_X86_64_64//.rodata
/*1a:*/  0x48, 0x89, 0xc6, //mov//rsi,rax
/*1d:*/  0xbf, 0x01, 0x00, 0x00, 0x00, //mov//edi,0x1
/*22:*/  0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, //movabs//rax,0x0
/*29:*/  0x00, 0x00, 0x00, 
/*24:*/  //R_X86_64_64//write
/*1179:*/  0x48, 0x89, 0xfa, //mov  //rdx,rdi  
/*117c:*/  0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00, //mov  //rdi,0x1  	
/*2c:*/  0xff, 0xd0, //call//rax
/*2e:*/  0x90, //nop
/*2f:*/  0xc9, //leave
/*30:*/  0xc3, //ret
};

void fix_code(char* str){
    memcpy(my_code + 0x24, &write_ptr, 8);
    memcpy(my_code + 0x12, &str, 8);
}

int power(int n, int pwr){
    if (pwr == 0)
        return 1;

    for(int i = 1; i < pwr; i++)
        n *= n;

    return n;
}

void* make_str(int num, char* str){
    // uint64_t str;
    int digit;

    for (int i = 1; i >= 0; i--){
        int pwr = power(10, i);

        // printf("pwr: %d %d\n", i, pwr);
        if ((digit = num / pwr) > 0){
            // printf("digit: %d\n", digit);
            str[1 - i] = (0x30 + digit);
            // str += (0x30 + digit) << i * 8;
            num -= pwr * digit;
        }
    }
}

typedef void (*sighandler_t)(int);

sighandler_t make_signal_handler(int num){
    char str[3];
    str[2] = '\0';

    make_str(num, str);
    fix_code(str);

    const size_t len = sizeof(my_code);

    /* mmap a region for our code */
    void *p = mmap(NULL, len, PROT_READ|PROT_WRITE,  /* No PROT_EXEC */
            MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (p==MAP_FAILED) {
        fprintf(stderr, "mmap() failed\n");
        return NULL;
    }

    /* Copy it in (still not executable) */
    memcpy(p, my_code, len);

    /* Now make it execute-only */
    if (mprotect(p, len, PROT_EXEC) < 0) {
        fprintf(stderr, "mprotect failed to mark exec-only\n");
        return NULL;
    }

    void (*func)(int) = (void(*)(int))p;

    return signal(num, func);
}

int main(void)
{
    sighandler_t old_2 = make_signal_handler(2);
    sighandler_t old_12 = make_signal_handler(12);

    return 0;
}