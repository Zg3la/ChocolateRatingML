import pandas as pd
import joblib
from django.shortcuts import render
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "choco_app" / "ml_model.pkl"


DATASET_PATH = BASE_DIR / "dataset" / "flavorsofcocoa.csv"

model = joblib.load(MODEL_PATH)


df = pd.read_csv(DATASET_PATH, encoding="cp1252")


companies = sorted(df["Company (Manufacturer)"].dropna().unique())
locations = sorted(df["Company Location"].dropna().unique())
bean_origins = sorted(df["Country of Bean Origin"].dropna().unique())


bar_names = sorted(df["Specific Bean Origin or Bar Name"].dropna().unique())
characteristics_list = sorted(df["Most Memorable Characteristics"].dropna().unique())
ingredients_list = sorted(df["Ingredients"].dropna().unique())


def predict_chocolate(request):
    rating = None

    if request.method == "POST":
        company = request.POST.get("company")
        cocoa_percent = float(request.POST.get("cocoa_percent"))
        company_location = request.POST.get("company_location")
        bean_origin = request.POST.get("bean_origin")

        bar_name = request.POST.get("bar_name", "")
        ingredients = request.POST.get("ingredients", "")
        characteristics = request.POST.get("characteristics", "")

       
        text_all = f"{bar_name} {ingredients} {characteristics}".strip()

        X = pd.DataFrame([{
            "company": company,
            "cocoa_percent": cocoa_percent,
            "company_location": company_location,
            "bean_origin": bean_origin,
            "text_all": text_all,
        }])

        rating = float(model.predict(X)[0])

    context = {
        "companies": companies,
        "locations": locations,
        "bean_origins": bean_origins,
        "bar_names": bar_names,

        
        "ingredients_list": ingredients_list,
        "characteristics_list": characteristics_list,

        "rating": rating,
    }

    return render(request, "predict.html", context)