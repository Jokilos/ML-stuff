cd ~/l*
sed -i '925i printk("Hello, ZSO!");' init/main.c 
# put the command in line 925, just before setup_arch()
# if '#ifdef CONFIG_X86_32' fails the first call in that 
# function is printk so it should work by then
make -j 16
sudo make install -j 16 
sudo reboot
