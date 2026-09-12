# Lab semana 4 — De la web a una API

Predicción de si una temporada de un equipo de la NHL fue "ganadora" (win_pct > 0.5)
a partir de gf (goles a favor), ga (goles en contra) y dif (diferencia de goles).
Datos bajados de [scrapethissite.com/pages/forms](https://www.scrapethissite.com/pages/forms/?page_num=1&per_page=100)
(582 filas, 6 páginas de 100 equipos).

Todo el flujo (scraping → dataset → modelo → API) está en [pipeline.py](pipeline.py),
se ejecuta con:

powershell
uv run python pipeline.py


Ese script deja escritos equipos.csv, equipos.parquet, modelos/modelo.joblib,
servir.py y Dockerfile, listos para construir la imagen de Docker.

## 1 · CSV vs Parquet — tamaños


csv        28.4 KB
parquet    12.8 KB


**Anota los dos tamaños. ¿Te sorprende cuál gana?**

Gana Parquet, y con bastante margen (menos de la mitad del peso del CSV). No me
sorprende: Parquet es un formato binario columnar, guarda cada columna con su
tipo de dato ya definido (enteros, floats) en vez de repetir todo como texto
plano separado por comas, y además comprime. El CSV en cambio escribe cada
número como caracteres ("300", "0.67"...), así que gasta más bytes por el
simple hecho de ser texto legible por humanos.

## 2 · Balance de clases


ganadora
False    0.65
True     0.35

diferencia media de goles en cada grupo:
ganadora
False   -21.3
True     40.2


El 65 % de las temporadas no son ganadoras y el 35 % sí: las clases están
desbalanceadas. La buena noticia es que dif sí separa bien los dos grupos
(en promedio -21 goles contra +40), lo que anticipa que el modelo real va a
poder aprender algo útil con esta variable.

## 3 · Modelo tonto vs. modelo real

| Modelo       | Exactitud | F1 (clase True) |
|--------------|-----------|--------------------|
| Tonto (baseline) | 0.651 | 0.000 |
| Regresión logística (el mío) | 0.811 | 0.723 |

**Compara la exactitud y el F1 de los dos modelos. ¿Qué hace mal el modelo
tonto que mirar únicamente la exactitud podría ocultar?**

Si solo mirara la exactitud, el modelo tonto parecería razonable: acierta el
65 % de las veces sin hacer ningún esfuerzo, y mi modelo "solo" sube a 81 %,
una mejora que a simple vista no parece enorme. El problema es que el tonto
consigue ese 65 % **repitiendo siempre False**: nunca predice una sola
temporada ganadora. Su recall para la clase True es 0 (no encuentra
ninguna) y por lo tanto su F1 también es 0. La exactitud alta lo esconde
porque hay muchas más temporadas perdedoras que ganadoras (65/35), así que
"decir siempre no" ya acierta la mayoría por pura suerte del desbalance. Mi
modelo, en cambio, sí detecta el 70 % de las temporadas ganadoras reales
(recall 0.70) con una precisión de 0.74, lo que da un F1 de 0.723 — la
diferencia real de que el modelo aprendió algo, en vez de solo repetir la
clase mayoritaria.

## 4 · Por qué se guarda el Pipeline completo y no solo la regresión logística

StandardScaler calculó, con los datos de entrenamiento, la media y la
desviación estándar de gf, ga y dif, y LogisticRegression aprendió sus
coeficientes **asumiendo que las entradas ya vienen escaladas con esos mismos
números**. Si guardara solo el clasificador y no el pipeline entero, la API
recibiría los goles en su escala original (números tipo 300, 250) en lugar de
los valores escalados (tipo -0.2, 1.3) con los que el modelo fue entrenado.
El clasificador no lanzaría ningún error porque técnicamente sigue recibiendo
tres números, pero las predicciones serían incorrectas sin ningún aviso, ya
que la regresión logística interpretaría esas escalas distintas como si
fueran patrones reales. Guardar el Pipeline completo garantiza que
cualquier dato nuevo pase por **exactamente** las mismas dos etapas
(escalado + clasificación) que se usaron en el entrenamiento.

## 5 · Docker sin -v

Corrí el contenedor sin el flag -v (sin montar la carpeta modelos/ local)
y el contenedor se cae inmediatamente al arrancar: no llega a exponer el
puerto 8000 y termina (docker ps ya no lo lista). La causa es que la imagen
de Docker solo copia servir.py (ver Dockerfile), nunca el archivo
modelos/modelo.joblib — ese archivo vive fuera de la imagen, en mi
computadora. Sin el -v "$PWD/modelos:/app/modelos", dentro del contenedor
la carpeta modelos/ no existe, así que la línea joblib.load("modelos/modelo.joblib")
lanza un FileNotFoundError al importar servir.py, lo que hace fallar el
arranque de uvicorn y el contenedor se cierra solo. El -v es justamente
lo que conecta el modelo entrenado (que vive en mi disco) con el código que
corre aislado dentro del contenedor.

## Captura de la API

GET /predecir?gf=300&ga=250 responde 200 OK:

json
{
  "ganadora": true,
  "probabilidad": 0.895
}


Parámetros enviados (gf=300, ga=250) y el curl que genera Swagger:

![Parámetros de la petición](capturas/api_predecir_parametros.png)

Respuesta del servidor (código 200, cuerpo y headers):

![Respuesta de la API](capturas/api_predecir_respuesta.png)

Esquemas de respuesta documentados por FastAPI (200 y 422):

![Esquemas de respuesta](capturas/api_predecir_esquemas.png)

## Cómo reproducir

powershell
uv sync
uv run python pipeline.py
docker build -t nhl-api .
docker run -p 8000:8000 -v "${PWD}/modelos:/app/modelos" nhl-api
# abrir http://localhost:8000/docs

