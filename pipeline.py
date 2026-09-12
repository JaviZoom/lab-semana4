"""Actividad "De la web a una API" - lab semana 4.

Baja las estadisticas de equipos de la NHL, entrena un modelo que predice si
una temporada fue ganadora, guarda el pipeline entrenado y genera los
archivos que necesita Docker (servir.py y Dockerfile) para exponerlo como API.

Ejecutar con: uv run python pipeline.py
"""

import os
import pathlib
import time

import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

URL = "https://www.scrapethissite.com/pages/forms/?page_num={}&per_page=100"
COLS = [
    "equipo", "anio", "victorias", "derrotas", "ot",
    "win_pct", "gf", "ga", "dif",
]


def leer_pagina(n: int) -> list[dict]:
    """Devuelve las filas de una pagina como lista de diccionarios."""
    sopa = BeautifulSoup(
        requests.get(URL.format(n), timeout=20).text, "html.parser"
    )
    filas = []
    for tr in sopa.select("tr.team"):
        celdas = [td.text.strip() for td in tr.select("td")]
        filas.append(dict(zip(COLS, celdas)))
    return filas


def bajar_datos() -> pd.DataFrame:
    respaldo = pathlib.Path("equipos_respaldo.csv")
    if respaldo.exists():
        print("Usando respaldo local equipos_respaldo.csv")
        return pd.read_csv(respaldo)

    datos = []
    for n in range(1, 7):  # 6 paginas x 100 filas
        datos += leer_pagina(n)
        print(f"pagina {n}", end="  ")
        time.sleep(0.5)
    print("\n", len(datos), "filas")
    return pd.DataFrame(datos)


def main() -> None:
    # 1. Bajar los datos
    equipos = bajar_datos()

    for c in ["anio", "victorias", "derrotas", "ot", "gf", "ga", "dif"]:
        equipos[c] = pd.to_numeric(equipos[c], errors="coerce")
    equipos["win_pct"] = pd.to_numeric(equipos["win_pct"], errors="coerce")

    equipos.to_csv("equipos.csv", index=False)
    equipos.to_parquet("equipos.parquet", index=False)  # necesita pyarrow

    csv_kb = os.path.getsize("equipos.csv") / 1024
    parquet_kb = os.path.getsize("equipos.parquet") / 1024
    print(f"csv      {csv_kb:6.1f} KB")
    print(f"parquet  {parquet_kb:6.1f} KB")

    # 2. La variable objetivo
    equipos["ganadora"] = equipos["win_pct"] > 0.5
    print(equipos["ganadora"].value_counts(normalize=True).round(2))
    print("diferencia media de goles en cada grupo:")
    print(equipos.groupby("ganadora")["dif"].mean().round(1))

    # 3. X e y
    X = equipos[["gf", "ga", "dif"]]
    y = equipos["ganadora"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    print(f"entrenamiento: {len(X_tr)} filas")
    print(f"prueba:        {len(X_te)} filas")

    # 4. Modelo tonto (baseline)
    tonto = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    p = tonto.predict(X_te)
    print(
        f"MODELO TONTO  exactitud {accuracy_score(y_te, p):.3f}"
        f"  F1 {f1_score(y_te, p):.3f}"
    )

    # 5. Modelo de verdad
    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    modelo.fit(X_tr, y_tr)
    pred = modelo.predict(X_te)
    print(
        f"TU MODELO     exactitud {accuracy_score(y_te, pred):.3f}"
        f"  F1 {f1_score(y_te, pred):.3f}"
    )

    print("--- MODELO TONTO")
    print(classification_report(y_te, p, zero_division=0))
    print("--- TU MODELO")
    print(classification_report(y_te, pred, zero_division=0))

    ConfusionMatrixDisplay.from_estimator(
        modelo, X_te, y_te, display_labels=["perdedora", "ganadora"]
    ).figure_.savefig("matriz_confusion.png")

    # 6. Sacarlo de la computadora
    pathlib.Path("modelos").mkdir(exist_ok=True)
    joblib.dump(modelo, "modelos/modelo.joblib")  # el Pipeline ENTERO
    print("modelos/modelo.joblib guardado")

    servir = [
        "import joblib",
        "from fastapi import FastAPI",
        "",
        'modelo = joblib.load("modelos/modelo.joblib")',
        'app = FastAPI(title="Temporada ganadora")',
        "",
        '@app.get("/predecir")',
        "def predecir(gf: int, ga: int):",
        "    x = [[gf, ga, gf - ga]]",
        '    return {"ganadora": bool(modelo.predict(x)[0]),',
        '            "probabilidad": round(float(modelo.predict_proba(x)[0][1]), 3)}',
    ]

    imagen = [
        "FROM python:3.12-slim",
        "WORKDIR /app",
        "RUN pip install --no-cache-dir fastapi uvicorn scikit-learn joblib",
        "COPY servir.py .",
        'CMD ["uvicorn", "servir:app", "--host", "0.0.0.0", "--port", "8000"]',
    ]

    pathlib.Path("servir.py").write_text("\n".join(servir) + "\n", encoding="utf-8")
    pathlib.Path("Dockerfile").write_text("\n".join(imagen) + "\n", encoding="utf-8")
    print("servir.py y Dockerfile escritos")


if __name__ == "__main__":
    main()
