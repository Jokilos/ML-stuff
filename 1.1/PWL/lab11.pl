% Reprezentacja grafów (skierowanych, nieskierowanych, etykietowanych itp.)
%   a) termowa: listy krawędzi, listy/macierze sąsiedztwa
%   b) klauzulowa
%       zbiór klauzul unarnych = zbiór krawędzi, zbiór list sąsiedztwa itd.

% 1. Rozważmy grafy typu DAG (skierowane acykliczne).
%    Zdefiniuj predykaty:
%     a) connect(A, B), connect(Graf, A, B) <==> istnieje ścieżka z A do B,
%         tzn. istnieje niepusty ciąg krawędzi prowadzących z A do B
%        (dwie wersje: dwie reprezentacje grafu - termowa i klauzulowa, zbiory krawędzi)

edges([e(1,2), e(2,3), e(3,4), e(3,5)]).

is_edge(A, B) :- edges(EList), member(e(A, B), EList).
connect(A, B) :- is_edge(A, B).
connect(A, B) :- is_edge(A, X), connect(X, B).

edge(1,2).
edge(2,3).
edge(3,4).
edge(3,5).
edge(4,2).

connectP(A, B) :- edge(A, B).
connectP(A, B) :- edge(A, X), connectP(X, B).

%     b) path(A,B,P) <==> P = ścieżka z A do B (jw.),
%        tzn. P = [A, ..., B], czyli lista wierzchołków kolejnych krawędzi.

path_aux(A, B, [e(A,B)]) :- edge(A, B).
path_aux(A, B, [e(A, X) | L]) :- edge(A, X), path_aux(X, B, L).

path(A, B, [A, B]):- edge(A, B).
path(A, B, [A, X | L]) :- edge(A, X), path(X, B, [X | L]).

% 2. Grafy skierowane (cykliczne).
%    Zdefiniuj predykat:   pathC(A,B,P) <==> P ścieżka z A do B (jw.),

pathC(A, B, _V, [A, B]):- edge(A, B).

pathC(A, B, V, [A | P]) :-
    edge(A, C),
    \+ member(C, V),
    pathC(C, B, [A | V], P).

% można zainicjalizować odwiedzone dodając A
pathC(A, B, P) :- pathC(A, B, [], P).

% 3. Grafy nieskierowane.
%    Zdefiniuj predykat euler/k (k >= 1) odnoszący sukces wtw, gdy podany
%    graf jest grafem Eulera i parametr P jest ścieżką Eulera w tym grafie
%    (tj. ścieżką przechodzącą przez każdą krawędź tego grafu dokładnie raz).

% 4. Graf skierowany acykliczny reprezentujemy za pomocą klauzul postaci:
%         graf(ListaWierzchołków) oraz
%         sasiedzi(W, ListaSąsiadówW),
% ListaSąsiadówW jest zbiorem (listą bez powtórzeń) wszystkich
% wierzchołków, do których prowadzi krawędź z wierzchołka W.

% Zdefiniuj predykat
%           odległe(W1, W2, Odległości),
% który odnosi sukces wtw, gdy Odległości jest uporządkowaną niemalejąco
% listą wszystkich odległości z wierzchołka W1 od wierzchołka W2.

% odległe(a, d, K)
% K = [1,2,2],

% odległe(b, C, L)
% C = b, L = [0] oraz C = d, L = [1].

odejmij1([], []).
odejmij1([X | L1], [Y | L2]) :-
    Y is X - 1,
    odejmij1(L1, L2).

graf([a,b,c,d]).
sasiedzi(a, [b,c,d]).
sasiedzi(b, [d]).
sasiedzi(c, [d]).
sasiedzi(d, []).

kraw(A, B) :-
    sasiedzi(A, L),
    member(B, L).

pathL(A, A, 0).

% pathL(A, B, s(0)) :-
%     kraw(A, B).

pathL(A, B, s(L)) :- 
    kraw(A, X),
    path(X, B, L).
    
% odlegle(A, A, L)