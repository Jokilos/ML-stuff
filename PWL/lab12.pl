% Drzewa otwarte
% -------------

% Zdefiniuj predykaty:

%   a) insertBST(Elem, DrzewoOtwarte)
%       (wstawienie Elem do drzewa BST - drzewa otwartego)
%       (3 wersje procedury: Elem jest (?) w drzewie)

% insertBST(E, tree(_, E, _)) :- !

insertBST(E, X) :- 
    var(X), !,
    X = tree(_X1, E, _X2).

insertBST(E, tree(TL, E1, _TR)) :-
    E1 > E, !,
    insertBST(E, TL).

insertBST(E, tree(_TL, E1, TR)) :-
    E1 =< E,
    insertBST(E, TR).

%   b) closeD(DrzewoOtwarte)
%        (zamknięcie drzewa otwartego)

closeBST(T) :-
    var(T), !,
    T = nil.

closeBST(tree(L, _, R)) :-
    closeBST(L),
    closeBST(R).

%   c) createBST(Lista, DrzewoBST-Zamknięte)

createBST([], T) :-
    closeBST(T).

createBST([X | L], T) :-
    insertBST(X, T),
    createBST(L, T).

%   d) sortBST(Lista, Posortowana)

% readBST(nil, _).

% readBST(tree(LT, X, RT), L2) :-
%     readBST(LT, L1),
%     L2 = [X | L1],
%     readBST(RT, L2).

% Listy różnicowe
% -------------------------

%  1. Flaga polska
%        (i) akumulator - lista różnicowa
%        (ii) parametr wyjściowy  - lista różnicowa

?- op(500, xfx, --).

flag([], []).

flag([X | L], F) :-
    flag([X | L], B, E, X),
    append(B, E, F).

flag([], [], [], _).

flag([X | L], [X | A], F, P) :-
    X = P,
    flag(L, A, F, P).
    
flag([X | L], A, [X | F], P) :-
    X \= P,
    flag(L, A, F, P).

%flag([b,c,b,c,c], [c], [b,b])

flag1([], []).
flag1([X | L], F) :-
    flag1(L, [], F, X).

flag1([], A, A, _).

flag1([X | L], A, [X | F], X) :- flag1(L, A, F, X).

flag1([Y | L], A, F, X) :- 
    Y \= X,
    flag1(L, [Y | A], F, X).

% flag(L, F) :- flag(L, X--X, F).

% flag([E | L], X--X, F) :-
%     flag(L, [E | X]--X, F).



% 2. Flaga holenderska (Red-White-Blue)
%        flaga_holenderska(+ListaRWB, ?Flaga)

%     Warianty:
%         a) 3 akumulatory - listy różnicowe
%         b) 2 listy różnicowe
%         c) 1 lista różnicowa
%         d) inne

% 3. Predykat wszerz(+DrzewoBinarne, ?ListaWierzchołkówWszerz).
% Klasyczny algorytm przejścia po drzewie wszerz z użyciem listy różnicowej do reprezentacji kolejki. 

    



