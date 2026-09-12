import joblib
from fastapi import FastAPI

modelo = joblib.load("modelos/modelo.joblib")
app = FastAPI(title="Temporada ganadora")

@app.get("/predecir")
def predecir(gf: int, ga: int):
    x = [[gf, ga, gf - ga]]
    return {"ganadora": bool(modelo.predict(x)[0]),
            "probabilidad": round(float(modelo.predict_proba(x)[0][1]), 3)}
