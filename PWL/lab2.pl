%  nat(x) == x jest liczba naturalną

nat(z).
nat(s(X)) :- nat(X).

%  plus(x,y,z) == x + y = z (w dziedzinie N)

mplus(X, z, X).
mplus(X, s(Y), s(Z)) :- mplus(X, Y, Z).

%  minus(x, y, z) == x - y = z

mminus(X, Y, Z) :- mplus(Y, Z, X).

%  fib(k, x) == x jest k-tą liczbą Fibonacciego

fib(z,z).

fib(s(z), s(z)).

% fib(s(s(K)), N) :-
%     fib(s(K), F2),
%     fib(K, F1),
%     mplus(F1, F2, N).

fib(s(s(K)), N) :- fib(s(s(K)), N, _).

fib(s(z), s(z), z).

fib(s(K), X, Y) :-
    mplus(X, Y, X1),
    fib(K, X1, X).

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

%  mod(x, y, z) == x modulo y = z

nmod(Y, X, Y) :-
    gt(X, Y).

nmod(Y, X, Z) :-
    gt(Y, X),
    plusa(Y1, X, Y),
    nmod(Y1, X, Z).

%  exp(n, x, z) == x^n = z
%  silnia(n, s) == s = n!
%  nwd(x, y, z) == z = największy wspólny dzielnik x i y 