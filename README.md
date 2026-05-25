# Práctica 1 — Optimización de preprocesamiento de imágenes con paralelismo

**Materia:** Cómputo de Alto Desempeño  
**Alumnos:** Jessica Melani Romero Lora  
**Docente:** Dr. Ernesto García Amaro  
**Institución:** Universidad Politécnica Metropolitana de Hidalgo (UPMH)  
**Fecha:** Mayo 2026

---

## Descripción

Pipeline de preprocesamiento de imágenes optimizado para alto desempeño. Parte de un código base secuencial con ciclos anidados y lo transforma mediante cuatro estrategias concretas: vectorización NumPy, eliminación de contención en escritura, paralelismo multiproceso y análisis de desbalance de carga.

El dataset consiste en imágenes JPG organizadas en cuatro clases (A, B, C, D) con distribución desigual: 363, 922, 990 y 1 388 imágenes respectivamente.

---

## Estructura del repositorio

```
.
├── practica1_optimizacion_v2.ipynb   # Notebook principal (código optimizado)
├── output/
│   ├── clase_A.csv                   # CSV por clase (modo paralelo)
│   ├── clase_B.csv
│   ├── clase_C.csv
│   ├── clase_D.csv
│   ├── dataset_final.csv             # Dataset unificado (3 663 filas × 257 columnas)
│   ├── comparacion_secuencial_paralelo.png
│   ├── tiempo_por_clase.png
│   └── secuencial/
│       ├── clase_A.csv               # CSV por clase (modo secuencial)
│       ├── clase_B.csv
│       ├── clase_C.csv
│       └── clase_D.csv
└── README.md
```

---

## Actividades

### Actividad 1 — Optimización de la rutina de procesamiento

El código original extraía píxeles con dos ciclos `for` anidados (256 iteraciones Python por imagen, ~920 000 en total). La versión optimizada aplica tres cambios en cascada:

| Cambio | Beneficio |
|--------|-----------|
| `cv2.IMREAD_GRAYSCALE` | Elimina dos conversiones de espacio de color (BGR → RGB → Gris) |
| `ndarray.flatten()` | Sustituye los ciclos anidados por una operación C compilada |
| Escritura por lotes con `pandas` | Reduce operaciones de I/O de N (una por imagen) a 1 (una por clase) |

### Actividad 2 — Un CSV por clase con núcleo dedicado

En el código original todos los procesos escribían en el mismo archivo en modo `append`, generando contención: el sistema operativo serializa los accesos a disco. La solución asigna un archivo exclusivo por clase (`clase_A.csv`, …) para que los cuatro núcleos trabajen de forma completamente independiente. La fusión final se ejecuta una sola vez en el proceso principal mediante `pd.concat()`.

### Actividad 3 — Comparación secuencial vs paralelo

| Modalidad | Tiempo |
|-----------|--------|
| Secuencial | 9.32 s |
| Paralelo   | 4.99 s |
| **Speedup**| **1.87×** |

El tiempo paralelo está acotado por la clase más lenta (D, 1 388 imágenes), conforme a la Ley de Amdahl.

### Actividad 4 — Tiempo por clase individual

| Clase | Imágenes | Tiempo (s) | Throughput |
|-------|----------|------------|------------|
| A     | 363      | 1.86       | 195.0 img/s |
| B     | 922      | 3.37       | 273.3 img/s |
| C     | 990      | 3.53       | 280.5 img/s |
| D     | 1 388    | 4.76       | 291.9 img/s |

La correlación directa entre número de imágenes y tiempo confirma comportamiento O(n) de la rutina vectorizada. La Clase D actúa como cuello de botella estructural del modo paralelo.

---

## Requisitos

```
python >= 3.10
opencv-python
numpy
pandas
matplotlib
```

Instalación:

```bash
pip install opencv-python numpy pandas matplotlib
```

---

## Uso

1. Ajustar la variable `DATASET_ROOT` en la celda de configuración del notebook para apuntar al directorio del dataset local.
2. Ejecutar todas las celdas en orden (`Run All`).
3. Los archivos CSV y las gráficas se generan automáticamente en `output/`.

---

## Resultados

El dataset final unificado contiene **3 663 filas × 257 columnas** (256 valores de píxel en escala de grises + etiqueta de clase). El speedup empírico de **1.87×** refleja el desbalance de carga entre clases: con particionado estático (una clase por núcleo), el tiempo total está dominado por la clase de mayor volumen. Un esquema de particionado dinámico (*work-stealing*) permitiría acercarse al límite teórico de 4× con cuatro núcleos.
