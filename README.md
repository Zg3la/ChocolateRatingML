RUAP Projekt – Predviđanje kvalitete čokolade
Instalacija potrebnih paketa

Instalirajte sve potrebne pakete koristeći sljedeće naredbe:

pip install django
pip install pandas
pip install numpy
pip install scikit-learn
pip install joblib


Postavljanje virtualnog okruženja

Kreirajte virtualno okruženje za projekt:

python -m venv venv

Aktivirajte virtualno okruženje:
venv\Scripts\activate

Pokretanje Django projekta

Prije pokretanja servera potrebno je ući u direktorij Django projekta:
cd choco_site
Pokrenite razvojni server:
python manage.py runserver
Otvorite web preglednik i posjetite:
http://127.0.0.1:8000/
Na prikazanoj stranici moguće je unijeti podatke o čokoladi i dobiti predviđenu ocjenu kvalitete pomoću istreniranog modela strojnog učenja.
