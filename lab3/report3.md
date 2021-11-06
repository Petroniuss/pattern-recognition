## Zadanie 3 -Pierwszy i prawdopodobnie ostatni raz w życiu (trenujemy od podstaw konwolucyjną siećneuronową)

Patryk Wojtyczek

## Dane

Pierwszy z brzegu wczytany egzemplarz (klasa 9 - truck)

![](imgs/example-1.png)

Każde zdjęcie z wczytanych 60k należy do jednej z klas poniżej.

- 0	airplane
- 1	automobile
- 2	bird
- 3	cat
- 4	deer
- 5	dog
- 6	frog
- 7	horse
- 8	ship
- 9	truck

## One hot encoding

Do zamienienia etykiet na prawdopodobieństwa korzystam z wbudowaną w tf funkcję `one_hot`.

![](imgs/one-hot-impl.png)

Przykładowy output dla etykiety.

![](imgs/one-hot-example.png)

## Minimalna architektura


![](imgs/minimal_architecture_impl.png)

Samych wag do strojenia modelu nie jest dużo, tylko 1180.

Model (przed treningiem) zwraca wiele niskich prawdopodobieństw o podobnej wartości co wskazuje, że nie umie 
jeszcze rozpoznać co jest na zdjęciach.

![](imgs/minimal_architecture_optimizer_impl.png)

Średni czas trwania epoki to 10s zatem całość trwała około 25 minut. Model szkoliłem na CPU
(jednak szybko przekonałem się, że lepiej przerzucić obliczenia na Colaba :) )

### Przebieg szkolenia i uzyskane wyniki na zbiorze walidacyjnym

![](imgs/minimal_arch_accuracy.png)

![](imgs/minimal_arch_loss.png)

Zwracane prawdopodobieństwa sugerują, że model nabrał pewnego "zrozumienia" co jest czym bo zazwyczaj zwraca 
jedno/dwa prawd które są wysokie a pozostałe są niskie.

\pagebreak

## Powiększanie sieci

### Zwiększenie liczby filtrów z 5 na 20

Liczba parametrów skoczyła do 7,390. Czas trenowania zmalał :) Ale to tylko dlatego, że przerzuciłem się na GPU.
Początkowo próbowałem szkolić na CPU z laptopa ale jedna epoka zajmowała 40s podczas gdy na GPU (colab) zajmowała 6s.

![](imgs/20_filters_acc.png)

![](imgs/20_filters_loss.png)

### Bloki konwolucyjne

Implementacja funkcji zwracającej cały blok - bardzo przydatny okazał się operator `*`.

![](imgs/block_impl.png)

Uruchomienie treningu takiego modelu dawało kiepskie wyniki - sieć się nie uczyła. Zmienienie aktywacji 
na ReLU znacząco poprawiło wyniki sieci. Czasowo - czas szkolenia epoki wzrósł do 8s.

Wyniki dwublokowej sieci z aktywacjami ReLU:

![](imgs/block_2_acc.png)

![](imgs/block_2_loss.png)

Możemy zaobserwować overfitting - accuracy na zbiorze treningowym się poprawia natomaist 
na zbiorze walidacyjnym się wypłaszcza a nawet zaczyna spadać.

### 4 bloki konwolucyjne

Przy 4 blokach konwolucyjnych osiągamy już całkiem pokaźną ilość parametrów: 459,950.
Czas uczenia epoki wzrósł do 12-13s.

![](imgs/block_4_acc.png)

![](imgs/block_4_loss.png)

Widać, że dalej mimo polepszania się wyników na zbiorze testowym accuracy na zbiorze walidacyjnym nie rośnie.

### Batch normalization

Dodanie warstwy batch normalization.

![](imgs/batch_normalization_impl.png)

Ma bardzo pozytywny wpływ na szybkość procesu uczenia - widać, że accuracy już od początkowych epok było rzędu 0.4 podczas kiedy
pozostałe wersje sieci potrzebowało wielu epok, żeby do takiego accuracy dojść. 

![](imgs/batch_normalization_acc.png)

![](imgs/batch_normalization_loss.png)


### Dropout

Dodanie warstwy z dropoutem.

![](imgs/dropout_impl.png)

Wyniki uzyskane po tej zmianie:

![](imgs/dropout_acc.png)

![](imgs/dropout_loss-3.png)

Sam czas uczenia nie zmienił się zauważalnie (nadal rzędu 12-13s na epokę). Widać natomiast, że udało się 
pokonać problem odbiegających wartości wyników uzyskanych na zbiorze walidacyjnym od tych na zbiorze treningowym.
Sieć bardziej się generalizuje, zamiast przesadnie dopasowywać się do danych, na których je trenujemy.

## GAP

Zmodyfikowanie sieci, aby przyjmowała obrazy o dowlonym rozmiarze nie wymagało ode mnie żadnej zmiany
(nigdzie tego nie zahardkodowałem).

Dodanie warstwy GAPu do ostatniego bloku zrealizowałem przez dodanie flagi.

![](imgs/gap_impl.png)

Wyniki i czas są podobne do poprzedniej konfiguracji:

![](imgs/gap_accuracy.png)

![](imgs/gap-loss.png)

\newpage

### Test na zdjęciu o innych rozmiarach

Test przeprowadziłem na zdjęciu samolotu (klasa 0)

![](imgs/plane.jpeg) 275 x 183

![](imgs/plane_scaled.jpeg) 41 x 27

Sieć dla takiego zdjęcia zwróciła 0.79 dla klasy 0 co wydaje się być bardzo dobrym wynikiem.
GAP może nam dać swego rodzaju niezależność co do takich drobnych odchyleń od rozdzielczości
zdjęcia na którym uczyliśmy sieć (I guess..).

### Wnioski

- ciężko przewidzieć jak sieć się zachowa (granted to była moja pierwsza) zanim spróbujemy ją wyszkolić.
- przez co uczenie sieci to żmudne zajęcie
- gpu radzi sobie o niebo lepiej od cpu w kontekście uczenia sieci.



\pagebreak
