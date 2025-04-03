#define BPF_NO_GLOBAL_DATA
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>


SEC("kprobe/do_nanosleep")
int handle(void *ctx)
{
    int pid = bpf_get_current_pid_tgid() >> 32;
    bpf_printk("PID %d is sleeping", pid);

    return 0;
}

char LICENSE[] SEC("license") = "GPL";
