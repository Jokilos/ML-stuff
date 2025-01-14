%   a) lista(X) wtw, gdy X jest (prologową) listą

lista([]).
lista([_ | _]).

%   d) ostatni(E, L) wtw, gdy E jest ostatnim elementem listy L

ost(X, [X]).
ost(X, [_, B | L]) :-
    ost(X, [B | L]).

%   e) element(E, L) wtw, gdy  E jest elementem L (member/2)

el(E, [E | _]).

el(E, [X | L]) :-
    E \= X,
    el(E, L).


%   f) intersection(L1, L2) wtw, gdy zbiory (listy) L1 i L2 nie sa rozłączne
%   g) scal(L1, L2, L3) wtw, gdy L1 o L2 = L3 (scalanie list; append/3)

%   a) podziel(Lista, NieParz, Parz) wtw, gdy NieParz (odp. Parz) jest
%      listą zawierającą wszystkie elementy listy Lista znajdujące się na
%      miejscach o nieparzystych (odp. parzystych) indeksach (zał. indeksujemy od 1)
%   b) wypisz(L) == czytelne wypisanie wszystkich elementów listy L
%        (elementy oddzielone przecinkami, na końcu kropka, lub CR,
%         info gdy lista pusta)
%   c) podlista(P, L) wtw, gdy P jest spójną podlistą L
%   d) podciag(P, L) wtw, gdy P jest (nie)spójną podlistą L (podciągiem)
%   e) srodek(E, L) wtw, gdy E jest środkowym elementem listy L  o nieparzystej długości,
%        czyli (k+1)-ym elementem, gdzie 2k+1 = długość listy L