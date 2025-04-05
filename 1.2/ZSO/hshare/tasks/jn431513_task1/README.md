## How to run the solution

1. Run the `setup.sh` script in order to install required packages. It also installs `gcc-aarch64-linux-gnu` which is required to run provided tests.
   Later the script creates virtual environment and installs `capstone` and `keystone-engine`.

2. In order to run the converter one can run the script `converter` with files provided as arguments or run `python conv/converter.py` in a similar fashion. 

3. In order to run the provided tests one can run the `run_tests.sh` script. In order for it to work i added a `converter` file in `z1_test`.

## Few words about the solution

I completed my solution using python.

Each important element of the file was described and kept in an object of a class corresponding to its name. For example all relocation entries are kept using objects of class `Rela` which is defined in the file `rela.py`. Those classes also have `fields` parameter, which specifies how specific fields in a file section are called according to ELF-64 format. One can read and change those values using `object.get('field_name')` or `object.set('field_name', value)`. If there is only one of something (elf file, elf header) everything is kept in a static fields of a class.

To pack and unpack the data i mostly used python `struct` functionality. I didn't really use any external dependencies apart from `capstone` and `keystone-engine`.