# Zadanie 4 -Zalewamy wrzątkiem chińską zupkę, czyli tym razem nasza sieć przygotowana jest z użyciem gotowej bazy

Patryk Wojtyczek

## Dane

Wybrałem następujące klasy:

- Chicken: `hong kong chicken`
- Dumplings: `dumplings`
- Salmon: `salmon`

Do załadowania zbioru danych wykorzystałem  `tf.keras.preprocessing.image_dataset_from_directory`.
Do podziału zbioru danych na dwie części wykorzystuję parametr `seed` i `validation_split`.

![](imgs/data-loading-code.png)

Załadowane egzemplarze wyglądają następująco

![](imgs/data-example.png)


## Sieci konwolucyjne na bazie gotowej kostki rosołowej

### Sieć z poprzedniego laboratorium

Szkolona od zera sieć z poprzedniego laboratorium uzyskała następujące wyniki:

![](imgs/lab3-net-loss.png)

![](imgs/lab3-net-acc.png)

### Transfer Learning z wyszkoloną siecią z poprzedniego laboratorium

Aby zapewnić, żeby batch-normalization działało w trybie inferencji przekazujemy parametr `training=False` gdy
wołamy base_model. Zamrażamy cały model poprzez ustawienie flagi `base_model.trainable=False`.

![](imgs/lab3-transferlearning.png)

Po liczbie `trainable params` widać, że wagi tylko ostatniej warstwy mogą się zmieniać.

![](imgs/lab3-frozen.png)

Uzyskane wyniki

![](imgs/transfer-learning-lab3-loss.png)

![](imgs/transfer-learning-lab3-acc.png)

Faktycznie wygląda to jakby dla naszej sieci poprzednio wyuczone wagi to było za mało.

### Transfer Learning z modelem Xception

![](imgs/transfer-learning-xception-code.png)

I uzyskane wyniki, wystarczyło kilka epok aby sieć uzyskała bardzo dobry wynik na zbiorze
walidacyjnym.

![](imgs/xception-loss.png)

![](imgs/xception-acc.png)

\newpage

### Fine-tuning modelu

Rozmrażamy wagi modelu. Batch-normalization nadal działa w trybie inferencji ze względu na przekazany
wcześniej parametr `training=False`. Parametr uczenia `learning_rate` jest ustawiony na znacznie
niższą wartość niż poprzednio.

![](imgs/fine-tuning-code.png)

Uzyskane wyniki

![](imgs/fine-tuning-loss.png)


![](imgs/fine-tuning-acc.png)

Na wykresach nieszczególnie widać zmianę, ale udało się podnieść `accuracy` z .87 do .90.

\newpage

## No dobra, ale w sumie na jakiej podstawie taka sieć podejmuje decyzje?

Po wrzuceniu do sieci takiego obrazka (kurczaka)

![](images/Chicken/006_4fccfeeb.jpg)

Kilka pierwszych z brzegu aktywacji wygląda w taki sposób. Mają rozdzielczość 8x8 
ale przy rysowaniu zrobiłem interpolację `billinear`.

![](imgs/couple-activations.png)

Po narzuceniu heatmap na obrazek (i zwiększeniu rozdzielczości) wygląda to następująco.

![](imgs/activations_on_img.png)

\newpage

Kod do obliczania sumarycznej heatmapy:

![](imgs/heatmap-code.png)

I otrzymujemy

![](imgs/heatmap.png)

Co wygląda sensownie - sieć skupiła się głównie na środkowym fragmencie który istotnie 
mówi, że na zdjęciu jest kurczak.

Heatmapy dla pozostałych wybranych fotografii

![](imgs/correct_heatmaps.png)

Faktycznie widać, że sieć skupiała się na istotnych miejscach - poza drugim trzecim zdjęciem
w pierwszym rzędzie gdzie skupiła się na jakimś pobocznym fragmencie i źle sklasyfikowała dany obraz - jako pierogi
zamiast kurczaka.

\newpage

Do odnalezienia fotografii, z którymi sieć sobie nie poradziła, porównałem predykcje z rzeczywistymi wynikami.
Niepoprawnie sklasyfikowanych było 32. 

![](imgs/wrong-predictions.png)

Niektóre z nich raczej nie powinny się tu znaleźć - np. niedźwiedź trzymający rybę klasyfikowany
jak kurczak wydaje się być swego rodzaju edge-casem. Puszka pewnie była tylko w zbiorze walidacyjnym.
A co do pozostałych to faktycznie można zobaczyć na jakich miejscach sieć się skupiła choć dalej nie wiadomo czemu
taką a nie inną decyzję podjęła - np. dla rysunku z pierogami wygląda jakby wybrała istotne do podjęcia decyzji miejsca
natomiast uważa, że jest to łosoś. Jest też kilka takch zdjęć z którymi sam miałbym trudność powiedzieć co tam jest.

![](imgs/incorrect-hm1.png)

![](imgs/incorrect-hm2.png)

