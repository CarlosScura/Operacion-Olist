# Operación Olist: Caos, Clientes y Algoritmos Desatados

Proyecto realizado en la especialización en datos de CodePRO (Penguin Academy).

En este proyecto se resuelven 3 misiónes de las 5 que teniamos a elección, las misiones corresponden al dataset público de Olist, un marketplace de Brasil. En las misiónes hay que: Predecir qué clientes vuelven a comprar de la tienda, estimar cuántos días tarda una entrega y detectar reseñas negativas a partir de su texto. En cada misión hay que determinar cuál es el mejor modelo, y se utilizan Decision Tree, Random Forest, Gradient Boosting, SVM lineal, Naive Bayes y kNN, según la misión. Cada misión toca distintas tablas del dataset.

El dataset de Olist reúne nueve tablas: pedidos, clientes, productos, vendedores, reseñas, entre otras. Sobre ese material se resuelven tres misiones, cada uno con varios modelos comparados entre sí y contra un modelo de base.
 
## Metodología
 
En las tres misiones se siguió el mismo camino:
 
- **Separación de los datos:** 80 % para entrenamiento y 20 % para prueba. El conjunto de prueba se deja aparte y se usa una única vez, al final, con el modelo elegido.
- **Pipelines:** el preprocesamiento, imputación de nulos, escalado, One-Hot Encoding o TF-IDF según la misión, va dentro de un `Pipeline` de scikit-learn, para que se ajuste solo con los datos de entrenamiento y no se filtre información del conjunto de prueba.
- **Modelo de base:** cada misión se compara contra un modelo que no aprende nada (siempre responde lo más común o el promedio), para saber si los modelos realmente aportan.
- **Validación cruzada de 5 vueltas** sobre el entrenamiento, con las métricas de entrenamiento y de prueba lado a lado. La diferencia entre las dos (la brecha) sirve para detectar sobreajuste.
- **Métricas según el problema:** F1, Precision y Recall en clasificación; MAE, RMSE y R² en regresión.
## Misión 1: ¿qué clientes vuelven a comprar?
 
**Planteo.** Predecir si una persona vuelve a comprar, a partir de su primer pedido entregado. Se define "volvió" como haber hecho más de una compra entregada. De 93.358 personas, 2.801 volvieron (cerca del 3 %), así que las clases están muy desbalanceadas. Se usaron tres variables: días de entrega, estado del cliente y puntaje de la reseña.
 
**Modelos comparados:** Decision Tree, Random Forest, Gradient Boosting, kNN, Naive Bayes y SVM lineal, más dos modelos de base. La métrica principal fue el F1 de la clase "volvió".
 
| Modelo | F1 en validación |
|---|---|
| Decision Tree | 0,060 |
| Random Forest | 0,060 |
| SVM lineal | 0,059 |
| Base: siempre "volvió" | 0,058 |
| Naive Bayes | 0,037 |
| kNN | 0,001 |
| Gradient Boosting | 0,000 |
 
**Hallazgos.** Ningún modelo supera claramente a la base que responde "volvió" para todos. Con las tres variables usadas hay solo 4.251 combinaciones distintas de valores en 74.686 filas de entrenamiento, y los resultados sugieren que no alcanzan para distinguir a quien vuelve. Se eligió el SVM lineal, el de menor brecha entre entrenamiento y prueba, aclarando que la elección es provisional. En la evaluación final con el conjunto de prueba obtuvo un F1 de 0,058, igual al de la base.
 
## Misión 3: ¿cuántos días tarda una entrega?
 
**Planteo.** Predecir los días entre la compra y la entrega de 96.470 pedidos entregados. Se usaron 11 variables: ubicación (estado del cliente y del vendedor), categoría del producto, flete, peso y medidas del producto, el margen hasta la fecha límite de envío, el plazo prometido y el volumen de pedidos del vendedor.
 
**Modelos comparados:** Decision Tree (con varios límites), Random Forest y Gradient Boosting, contra un modelo que predice siempre el promedio.
 
| Modelo | MAE en validación (días) | RMSE | R² | Brecha de MAE |
|---|---|---|---|---|
| Base: promedio | 6,419 | 9,609 | 0,000 | 0,000 |
| Decision Tree (mín. 50 por hoja) | 5,216 | 8,458 | 0,225 | 0,322 |
| Random Forest | 5,019 | 8,234 | 0,266 | 0,205 |
| Gradient Boosting | 5,083 | 8,301 | 0,254 | 0,027 |
 
**Hallazgos.** Los tres modelos mejoran a la base. El Random Forest obtiene los mejores valores, aunque la ventaja sobre el Gradient Boosting es chica; el Gradient Boosting es el que menos sobreajusta. En la evaluación final con el conjunto de prueba, el Random Forest obtuvo un MAE de 4,967 días, un RMSE de 7,846 y un R² de 0,291. Es decir, se equivoca en promedio unos 5 días, y el modelo explica entre un cuarto y un tercio de la variación de los tiempos de entrega.
 
## Misión 5: ¿qué reseñas son negativas?
 
**Planteo.** Clasificar reseñas como negativas (1, 2 o 3 estrellas) o positivas (4 o 5 estrellas), a partir del texto en portugués. Se usaron las 42.706 reseñas que tienen título o comentario (de 99.224), juntando ambos en un solo texto; 14.638 son negativas y 28.068 positivas. El texto se convierte en números con TF-IDF.
 
**Modelos comparados:** Naive Bayes y SVM lineal, contra un modelo que siempre responde "positiva". La métrica principal fue el F1 de la clase negativa.
 
| Modelo | F1 entrenamiento | F1 validación | Brecha de F1 |
|---|---|---|---|
| Base: siempre "positiva" | 0,000 | 0,000 | 0,000 |
| Naive Bayes | 0,860 | 0,835 | 0,025 |
| SVM lineal | 0,926 | 0,845 | 0,081 |
 
**Hallazgos.** Los dos modelos superan ampliamente a la base. El SVM lineal tiene un F1 apenas más alto, pero sobreajusta bastante más, por lo que se eligió el Naive Bayes. En la evaluación final con el conjunto de prueba obtuvo un F1 de 0,842, una Precision de 0,852 y un Recall de 0,833.
 
## Estructura del repositorio
 
```
Operacion-Olist/
├── operacion_olist.ipynb   # notebook con las tres misiones
├── requirements.txt        # librerías necesarias
├── README.md
└── archive/                # datos de Olist (no se versionan)
```
 
## Tecnologías
 
- Python 3.13
- pandas
- matplotlib
- scikit-learn
- Jupyter
## Cómo correrlo
 
1. Clonar el repositorio y crear el entorno virtual:
```bash
git clone https://github.com/CarlosScura/Operacion-Olist.git
cd Operacion-Olist
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
 
2. Descargar el dataset público de Olist desde Kaggle (*Brazilian E-Commerce Public Dataset by Olist*) y copiar los archivos CSV en una carpeta `archive/` en la raíz del proyecto.
3. Abrir `operacion_olist.ipynb` y ejecutar las celdas en orden.
## Autor
 
Carlos Alarcón Scura — Estudiante de Penguin Academy
