# Latent and Robust Multi-label Graph Convolutional Network (LaR-MLGCN)

ImplementaciÃ³n en PyTorch de **LaR-MLGCN**, un enfoque para clasificaciÃ³n multietiqueta en datos tabulares que modela explÃ­citamente dependencias entre etiquetas mediante una estructura de grafo.

## Resumen

La clasificaciÃ³n multietiqueta plantea el desafÃ­o crÃ­tico de modelar dependencias complejas entre categorÃ­as en espacios de salida de alta dimensionalidad. A pesar de los avances recientes, los enfoques tradicionales a menudo fallan al capturar estas correlaciones de manera explÃ­cita o carecen de la eficiencia necesaria para escalar.  

En este trabajo proponemos una arquitectura basada en Redes Neuronales de Grafos (GCN) que desplaza el nÃºcleo del aprendizaje hacia un espacio relacional de etiquetas. El enfoque emplea una GCN para generar dinÃ¡micamente clasificadores parametrizados, integrando informaciÃ³n semÃ¡ntica derivada de la topologÃ­a de coocurrencia de los datos, reforzada por una polÃ­tica de regularizaciÃ³n adaptativa sensible a la densidad.  

La evaluaciÃ³n exhaustiva sobre 19 conjuntos de datos de referencia demuestra que el modelo propuesto supera consistentemente al estado del arte, alcanzando los mejores rangos promedio en diferentes mÃ©tricas de rendimiento. Los resultados, validados estadÃ­sticamente, confirman que la integraciÃ³n de una estructura relacional explÃ­cita no solo maximiza la capacidad predictiva, sino que ofrece una alternativa estructuralmente mÃ¡s eficiente e interpretable que los mÃ©todos basados en comitÃ©s de clasificadores.

## Aportaciones Principales

- Modelado explÃ­cito de dependencias de etiquetas vÃ­a grafo de coocurrencia.
- GeneraciÃ³n de clasificadores por etiqueta mediante propagaciÃ³n en GCN.
- PolÃ­tica de regularizaciÃ³n adaptativa segÃºn densidad/grado del grafo.
- DiseÃ±o eficiente e interpretable para clasificaciÃ³n multietiqueta tabular.

## Arquitectura

La arquitectura combina dos rutas:

1. **Ruta de etiquetas (GCN)**  
   - Construye un grafo de etiquetas a partir de `Y_train`.
   - Propaga embeddings latentes de etiquetas y produce una matriz de clasificadores `W`.

2. **Ruta de datos (MLP)**  
   - Proyecta las variables de entrada `X` en un espacio latente `Phi`.

3. **PredicciÃ³n**  
   - Calcula logits como `Phi @ W.T` (opcionalmente con sesgo por clase).

## InstalaciÃ³n

Requisitos:

- Python 3.10+
- Dependencias definidas en `requirements.txt`

```bash
pip install -r requirements.txt
```

## Uso RÃ¡pido

```python
import numpy as np
from MLGNNClassifiers import LaRMLGCNClassifier

X_train = np.random.randn(200, 20).astype(np.float32)
Y_train = np.random.randint(0, 2, size=(200, 5)).astype(np.int64)
X_test = np.random.randn(50, 20).astype(np.float32)

model = LaRMLGCNClassifier(
    input_dim=20,
    num_labels=5,
    proj_dim=128,
    mlp_hidden_dims=(256,),
    label_embed_dim=32,
    gcn_hidden_dims=(128,),
    tau=0.4,
    p=0.2,
)

model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=True)
proba = model.predict_proba(X_test)              # (50, 5), rango [0, 1]
pred = model.predict(X_test, threshold=0.5)      # (50, 5), salida binaria
```

## Entrenamiento y EvaluaciÃ³n

Script principal de entrenamiento/predicciÃ³n:

```bash
python run_mlgcn.py <train_csv> <test_csv> <last_n_labels> <output_preds> --seed 42 --cv
```

Donde:

- `train_csv`: dataset de entrenamiento.
- `test_csv`: dataset de prueba.
- `last_n_labels`: nÃºmero de columnas finales que corresponden a etiquetas.
- `output_preds`: archivo de salida con predicciones binarias.
- `--cv`: activa bÃºsqueda por validaciÃ³n cruzada.

## Reproducibilidad

Pruebas unitarias e integraciÃ³n ligera:

```bash
python -m unittest test_mlgcn_classifier.py
```

En Windows con Anaconda:

```powershell
python.exe -m unittest test_mlgcn_classifier.py
```

## Estructura del Repositorio

- `MLGNNClassifiers.py`: implementaciÃ³n de `MLGCNClassifier`.
- `run_mlgcn.py`: pipeline de entrenamiento, ajuste de umbrales y predicciÃ³n.
- `evaluate_mlgcn_performance.py`: utilidades de evaluaciÃ³n.
- `test_mlgcn_classifier.py`: tests unitarios y de integraciÃ³n.
- `requirements.txt`: dependencias.

## Cita

Si este repositorio te resulta Ãºtil en investigaciÃ³n, por favor cita el trabajo asociado.  
Puedes reemplazar la entrada siguiente cuando tengas los metadatos finales:

```bibtex
@article{larmlgcn2026,
  title   = {Latent and Robust Multi-label Graph Convolutional Network},
  author  = {Perez, ...},
  journal = {Under review},
  year    = {2026}
}
```

## Licencia

Define aquÃ­ la licencia del proyecto (por ejemplo, `MIT`, `Apache-2.0` o `GPL-3.0`) y aÃ±ade el archivo `LICENSE` correspondiente.

