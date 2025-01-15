%  nat(x) == x jest liczba naturalną

nat(z).
nat(s(X)) :- nat(X).

%  plus(x,y,z) == x + y = z (w dziedzinie N)

mplus(X, z, X).
mplus(X, s(Y), s(Z)) :- mplus(X, Y, Z).

%  minus(x, y, z) == x - y = z

mminus(X, Y, Z) :- mplus(Y, Z, X).

%  fib(k, x) == x jest k-tą liczbą Fibonacciego

to_nat(0, z).

to_nat(X, s(Y)) :-
    X > 0,
    X1 is X - 1,
    to_nat(X1, Y).

from_nat(z, 0).

from_nat(s(X), Y) :-
    from_nat(X, Y1),
    Y is Y1 + 1.

fib(z,z).

fib(s(z), s(z)).

fib(s(s(K)), N) :- fib(s(s(K)), N, _).

fib(s(z), s(z), z).

fib(s(K), X1, X) :-
    fib(K, X, Y),
    mplus(X, Y, X1).

check_fibl(X, Y) :-
    to_nat(X, XN),
    fib(XN, YN),
    from_nat(YN, Y).

:- begin_tests(fibonacci).

test(1) :-
    check_fibl(5, F),
    F =:= 5.
test(2) :-
    check_fibl(10, F),
    F =:= 55.
test(3) :-
    to_nat(13, X),
    fib(Y, X),
    from_nat(Y, Y1),
    Y1 =:= 7.

:- end_tests(fibonacci).

%  even(n) == n jest liczbą parzystą

even(z).
even(s(s(z))) :- even(z).

%  odd(n)   == n jest liczbą nieparzystą

odd(s(E)) :- even(E).

%  razy(x, y, z) == x * y = z

nmul(z, _X, z).

nmul(s(X), Y, Z) :-
    nmul(X, Y, Z1),
    mplus(Y, Z1, Z).

%  le(x, y) ==  x <= y  (w dziedzinie liczb naturalnych)
%  lt(x, y) ==  x < y  (w dziedzinie liczb naturalnych)

gt(s(_), z).
gt(s(X), s(Y)) :- gt(X, Y).

eq(z, z).
eq(s(X), s(Y)) :- eq(X, Y).

lt(X, Y) :- gt(Y, X).

neq(X, Y) :- lt(X, Y).
neq(X, Y) :- gt(X, Y).

le(X, Y) :- eq(X, Y).
le(X, Y) :- lt(X, Y).

%  mod(x, y, z) == x modulo y = z

nmod(X, X, z).

nmod(Y, X, Y) :-
    gt(X, Y).

nmod(Y, X, Z) :-
    gt(Y, X),
    mplus(Y1, X, Y),
    nmod(Y1, X, Z).

%  exp(n, x, z) == x^n = z

exp(z, _, s(z)).
exp(s(z), X, X).

exp(s(N), X, R) :-
    gt(N, z),
    exp(N, X, R1),
    nmul(X, R1, R).

%  silnia(n, s) == s = n!

fac(z, s(z)).

fac(s(N), R) :-
    fac(N, R1),
    nmul(s(N), R1, R).

%  nwd(x, y, z) == z = największy wspólny dzielnik x i y

nwd(X, X, X).

nwd(X, Y, Z) :-
    gt(Y, X),
    nwd(Y, X, Z).

nwd(X, Y, Y) :-
    gt(X, Y),
    nmod(X, Y, z).

nwd(X, Y, R) :-
    gt(X, Y),
    nmod(X, Y, R1),
    neq(R1, z),
    nwd(Y, R1, R).
