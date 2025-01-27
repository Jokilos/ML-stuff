for i in $(seq -w 1 3); do
    wget --user warsaw --password 7gCsTQwVrpTVDnEk "https://www.mimuw.edu.pl/~iwanicki/courses/ds/2024/labs/LA$i/dsassignment$i.tgz" -O "assignment$i.tgz"
    # wget --user warsaw --password 7gCsTQwVrpTVDnEk "https://www.mimuw.edu.pl/~iwanicki/courses/ds/2024/labs/L$i/dslab$i.tgz" -O "dslab$i.tgz"
done
    

