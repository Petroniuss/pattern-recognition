## Zadanie 2 - Kwadraty bez trójkątów (za to z wysokim stężeniem konwolucji)

Patryk Wojtyczek

### Wejściowe dane

![](imgs/pipes.jpeg)

\pagebreak

### Konwersja zdjęcia do sklali szarości

Do inicjalizacji kernela wykorzystałem parametr kernel_initializer do którego można przekazać funkcję tworzącą kernel.
Wybrałem następujące wagi ```r=.299, g=.587, b=.114```

![](imgs/grayscale_code.png)

![](imgs/grayscale.png)

\pagebreak

### Gaussian Blur
Jako rozmiar kernela wybrałem n = 5.

![](imgs/blur_code.png)

![](imgs/blur.png)

\pagebreak

### Szukanie gradientu

Po zastosowaniu filtru Sobela otrzymujemy przybliżenie składowych gradientu w kierunku osi x i y.


W kierunku osi x.

![](imgs/gradient_x.png)

W kierunku osi y.

![](imgs/gradient_y.png)

Korzystając ze składowych gradientu możemy obliczyć wartość gradientu.

![](imgs/gradient_code.png)

![](imgs/gradient_intensity.png)

Następnie filtrujemy gradient używając ReLU

![](imgs/gradient_intensity_filtered.png)

## Pooling

Zmniejszamy tutaj rozdzielczość obrazu - pooling 8x8 jest o tyle przyjemny, że rozmiar jest wielokrotnością 8 więc upscaling stosowany później da nam ten sam rozmiar od którego zaczęliśmy. 
Zaimplementowałem jeszcze dodawanie paddingu (zer) do obrazu tak aby rozmiar był zawsze wielokrotnością `pool_size`.

![](imgs/pool_code.png)

\pagebreak

## Głosowanie

Teraz możemy zając się tworzeniem kerneli tworzących krawędzie kwadratu. Zaimplementowałem to poprzez stworzenie listy warstw z kernelami (z jednym wynikowym kanałem) różnych rozmiarów choć można to pewnie było zrobić z użyciem jednego
kernela (z wieloma wynikowymi kanałami).

![](imgs/voting.png)

\pagebreak
Wygląd jednego z kanałów (dla n=13).

![](imgs/y1.png)

![](imgs/y2.png)

![](imgs/y3.png)

\pagebreak

Nałożenie wszystkich kanałów na obraz.

![](imgs/result.png)

## Zastosowawnie stworzonej implementacji do innego obrazu
Próbowanie przeze mnie zdjęcie jest znacznie bardziej problematyczne bo zawiera dużo prostokątów (a my szukamy kwadratów) a kwadraty które występują na zdjęciu są stosunkowo nieregularne.

![](imgs/foo.jpeg)

\pagebreak
Wynik jest raczej kiepski, dałoby się pewnie osiągnąć lepszy wynik lepiej dostosowując parametry co jednak pokazuje słabą skalowalność metody - skoro dla każdego zdjęcia trzeba zmieniać parametry.

![](imgs/result2.png)

