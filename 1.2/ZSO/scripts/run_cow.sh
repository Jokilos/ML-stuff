qemu-system-x86_64                                              \
    -device virtio-scsi-pci,id=scsi0                            \
    -drive file=../data/IMG/zso2025_cow.qcow2,if=none,id=drive0 \
    -device scsi-hd,bus=scsi0.0,drive=drive0                    \
    -enable-kvm                                                 \
    -smp 16                                                      \
    -cpu host                                                   \
    -net nic,model=virtio                                       \
    -net user,hostfwd=tcp::2222-:22                             \
    -m 4G -device virtio-balloon                                \
    -fsdev local,id=hshare,path=../hshare/,security_model=none  \
    -device virtio-9p-pci,fsdev=hshare,mount_tag=hshare         \
    -chardev stdio,id=cons,signal=off                           \
    -device virtio-serial-pci                                   \
    -device virtconsole,chardev=cons                            \
    -display none                                               \

    # stty rows 236 cols 64 

    # -kernel <file> -append <options>
    # Runs the linux kernel directly from the given file with
    # the given options instead of going ahead of the standard
    # boot process. Sometimes useful.

    # -gdb tcp::<port>

    # Allows you to connect to qemu via gdb (the gdb command is
    # target remote localhost:<port>) and debug the kernel in
    # this way. Sometimes useful.

    # -S

    # In combination with the -gdb option, it causes qemu to
    # start in a suspended state, allowing you to set breakpoints
    # etc. by gdb before the system is started.

