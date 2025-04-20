## Instrukcja

Poniższy dokument stanowi zwięzłą instrukcję obsługi projektu służącego do klasyfikacji tekstu na danych AG_NEWS przy użyciu torchtext i prostego modeli EmbeddingBag. Znajdziesz tu opis struktury repozytorium, wymagania, sposób instalacji, uruchomienia treningu oraz testów, a także przykładowe fragmenty kodu i informacje o licencji.

## Opis projektu

Projekt realizuje klasyfikację tekstu na cztery kategorie (World, Sports, Business, Sci/Tec) z wykorzystaniem biblioteki torchtext oraz modelu opartego na EmbeddingBag i jednej warstwie liniowej
deepsense.ai
.

Przygotowany jest pełny pipeline: od pobrania i podziału danych AG_NEWS, przez budowę słownika i funkcję collate_fn, aż po pętlę treningową z obcinaniem gradientów i ewaluację
DataDrivenInvestor
.

Kod zawiera także funkcję pozwalającą na przewidywanie etykiet pojedynczych zdań oraz zapis najlepszego modelu (model.pth) w trakcie treningu
FreeCodeCamp
.  

## Wymagania

    Python ≥ 3.7

    torch ≥ 1.9.0

    torchtext ≥ 0.10.0

    numpy, pandas, matplotlib, scikit-learn

    tqdm

`pip install torch torchtext numpy pandas matplotlib scikit-learn tqdm`  

## Instalacja

    Sklonuj repozytorium:

`git clone https://github.com/Krzysiek-Mistrz/classyfing_documents_app.git`  
`cd classyfing_documents_app`  

Utwórz wirtualne środowisko i aktywuj je (opcjonalnie):

`python -m venv venv`  
`source venv/bin/activate`  # Linux/MacOS  
`venv\Scripts\activate`     # Windows  

## Struktura repozytorium

├── app.py  
├── LICENSE  
└── README.md  

## Uruchomienie

W terminalu wykonaj:

`python train.py`  

Skrypt automatycznie wykona predykcje dla zadanego zdania w kodzie **zmienna: sentence (na początku w kodzie (po #loading data & cr. classes))**:  

    Pobiera i przetwarza dane AG_NEWS.

    Buduje słownik i dataloadery z funkcją collate_fn.

    Tworzy model EmbeddingBag + Linear i trenuje go z CrossEntropyLoss i SGD.

    Zapisuje najlepszy model do pliku model.pth

## Ewaluacja i testowanie

Aby ocenić model na zestawie walidacyjnym/testowym, po prostu wykonaj skrypt. Jeżeli chcesz żeby twój model był najpierw trenowany to po prostu zamień miejscami kod pod *#model training* z kodem pod *#model & predicting & accuracy*. Wówczas przed predykcją dodatkowo wytrenujesz swój model (trwa koło 20min w zależności od komputera).

## Licencja

Projekt udostępniony na licencji GNU GPL 3.0. Szczegóły w pliku LICENSE.