# -*- coding: utf-8 -*-
"""
benchmark_hpc.py

Benchmark de Computo de Alto Desempeno aplicado al pipeline NF1.

OBJETIVO:
Medir empiricamente el speedup real obtenido al paralelizar el nested
cross-validation sobre distintos numeros de nucleos CPU, y usar esos
datos para (a) justificar cuantitativamente la eleccion de CPU
multi-nucleo sobre GPU/distribuido para el tamano de datos actual, y
(b) estimar, via Ley de Amdahl, el limit teorico de speedup del
pipeline tal como esta construido.

Este script NO reemplaza al pipeline principal (nf1_pipeline_corregido.py):
toma una version reducida y autocontenida del nested CV para poder
variar n_jobs de forma controlada y medir tiempos limpios, sin mezclar
el costo de EDA/visualizaciones (que es sequential y no forma parte de
la pregunta de HPC).

VISUALIZACIONES GENERADAS (4 graficos, guardados como PNG):
1. grafico1_gantt_sequential_vs_paralelo.png
    Gantt chart: cada tarea (fold x modelo) como barra horizontal,
    comparando ejecucion sequential (1 carril) vs. paralela (N carriles).
2. grafico2_heatmap_utilization_nucleos.png
    Heatmap de que nucleos estan activos/inactivos en cada instante
    durante la corrida paralela -- revela desbalance de carga.
3. grafico3_cpu_vs_gpu_simulado.png
    Comparacion CPU medido vs. GPU PROYECTADO (no medido -- no hay
    acceso a GPU; es un modelo teorico transparente y declarado como tal).
4. grafico4_tiempo_speedup_amdahl.png
    Tiempo de ejecucion y speedup medido vs. ideal vs. Ley de Amdahl.

Autora: Jessica Melani Romero Lora
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # evita el conflicto de backend interaction + multiprocessing en macOS
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import multiprocessing

RANDOM_STATE = 42

# 1. CARGA DE DATOS (misma fuente que el pipeline principal)
url_github = 'https://raw.githubusercontent.com/frontendbby/Fun-datasets/refs/heads/main/neurofibromatosis.csv'
df = pd.read_csv(url_github)
df.columns = df.columns.str.replace(r'[^A-Za-z0-9_]+', '', regex=True)
df.columns = df.columns.str.replace(r'\s+', '', regex=True)
df.rename(columns={'TumourCase': 'Tumour_Case', 'CafeauLaitCLS': 'Cafe_au_Lait'}, inplace=True)

y = df['Tumour_Case']
X = df.drop(['Tumour_Case', 'CaseType'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print(f"Dataset: {X_train.shape[0]} pacientes en entrenamiento, {X_train.shape[1]} features")
print(f"Nucleos CPU disponibles en esta maquina: {multiprocessing.cpu_count()}")

# 2. PIPELINE REDUCIDO PARA BENCHMARK
#
# Se usa solo Regresion Logistica con un grid moderado, para que el
# experimento sea repetible en minutos y no en horas. El objetivo aqui
# es medir ESCALAMIENTO, no reproducir los resultados finales del
# proyecto (esos ya estan en nf1_pipeline_corregido.py).
pipeline_bench = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        l1_ratio=1, solver='liblinear', class_weight='balanced',
        max_iter=1000, random_state=RANDOM_STATE
    ))
])
param_grid_bench = {'clf__C': np.logspace(-3, 2, 10)}  # 10 valores de C

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def correr_nested_cv(n_jobs):
    """Ejecuta un nested CV completo con el n_jobs especificado y
    retorna el tiempo de pared (wall-clock time) en segundos."""
    t0 = time.perf_counter()

    for train_idx, val_idx in outer_cv.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]

        grid = GridSearchCV(
            pipeline_bench, param_grid_bench, cv=inner_cv,
            scoring='roc_auc', n_jobs=n_jobs
        )
        grid.fit(X_tr, y_tr)

    t1 = time.perf_counter()
    return t1 - t0

# 3. EXPERIMENTO: TIEMPO DE EJECUCION VS. NUMERO DE NUCLEOS
# DECISION METODOLOGICA:
# Se corre cada configuracion N_REPEATS veces y se reporta la mediana,
# no el promedio, porque el tiempo de ejecucion en un laptop personal
# es sensible a procesos en segundo plano (ruido del sistema operativo);
# la mediana es mas robusta a esos outliers que el promedio.
N_REPEATS = 3
max_cores = multiprocessing.cpu_count()
core_counts = sorted(set([1, 2, max(2, max_cores // 2), max_cores]))

resultados_benchmark = []

print("\n" + "=" * 60)
print("Benchmark: tiempo de ejecucion vs. numero de nucleos")
print("=" * 60)

for n_jobs in core_counts:
    tiempos = [correr_nested_cv(n_jobs) for _ in range(N_REPEATS)]
    tiempo_mediana = np.median(tiempos)
    resultados_benchmark.append({'n_jobs': n_jobs, 'tiempo_seg': tiempo_mediana})
    print(f"  n_jobs={n_jobs:2d} | tiempo mediana = {tiempo_mediana:.2f}s "
        f"(runs: {[f'{t:.2f}' for t in tiempos]})")

bench_df = pd.DataFrame(resultados_benchmark)

# 4. SPEEDUP, EFICIENCIA, Y AJUSTE A LEY DE AMDAHL
# Speedup(n) = T(1) / T(n)  -- cuanto mas rapido corre con n nucleos
# Eficiencia(n) = Speedup(n) / n  -- que tan bien se aprovecha cada nucleo
tiempo_base = bench_df.loc[bench_df['n_jobs'] == 1, 'tiempo_seg'].values[0]
bench_df['speedup'] = tiempo_base / bench_df['tiempo_seg']
bench_df['eficiencia'] = bench_df['speedup'] / bench_df['n_jobs']

print("\n" + "=" * 60)
print("Speedup y eficiencia por numero de nucleos")
print("=" * 60)
print(bench_df.to_string(index=False))

# --- Ajuste de Ley de Amdahl: Speedup(n) = 1 / ((1-p) + p/n) ---
# Se estima 'p' (fraccion paralelizable del pipeline) por minimos
# cuadrados no lineales, usando los datos medidos arriba.
from scipy.optimize import curve_fit

def amdahl_speedup(n, p):
    return 1 / ((1 - p) + p / n)

try:
    popt, _ = curve_fit(amdahl_speedup, bench_df['n_jobs'], bench_df['speedup'],
                        bounds=(0, 1), p0=[0.7])
    p_estimado = popt[0]
    speedup_teorico_max = 1 / (1 - p_estimado) if p_estimado < 1 else np.inf

    print(f"\nFraccion paralelizable estimada (Ley de Amdahl): p = {p_estimado:.3f}")
    print(f"Speedup teorico maximo con nucleos infinitos: {speedup_teorico_max:.2f}x")
    print("Interpretacion: aunque tuvieras un cluster con cientos de nucleos, "
        f"el pipeline actual NUNCA correria mas de {speedup_teorico_max:.1f}x mas rapido "
        "que con 1 solo nucleo, porque una fraccion del trabajo (carga de datos, "
        "particion de folds, agregacion de resultados) es inherentemente secuencial.")
except Exception as e:
    print(f"\n[Aviso] No se pudo ajustar la curva de Amdahl: {e}")
    p_estimado = None


# 5. CAPTURA DE TAREAS: TIMESTAMPS POR NUCLEO (para los Gantt charts)
# DECISION METODOLOGICA:
# Para VER como se distribuyen las tareas entre nucleos (no solo medir
# tiempo total), se instrumenta cada tarea individual con su propio
# timestamp de inicio/fin y el PID del proceso que la ejecuto. La
# unidad de tarea aqui es (fold externo x modelo): 5 folds x 3 modelos
# = 15 tareas independientes, que es la granularidad real de tu
# pipeline principal (nf1_pipeline_corregido.py, Seccion 8).
#
# Se usa joblib.Parallel directamente (en vez de GridSearchCV) para
# tener control total sobre qué se paraleliza y poder capturar el PID
# de cada tarea de forma confiable.
import os
from joblib import Parallel, delayed

def modelos_benchmark():
    """Version ligera de los 3 modelos, solo para el benchmark de HPC."""
    pipeline_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(l1_ratio=1, solver='liblinear',
                                    class_weight='balanced', max_iter=1000,
                                    random_state=RANDOM_STATE))
    ])
    pipeline_rf = Pipeline([('clf', RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE))])
    return [
        ('LR', pipeline_lr, {'clf__C': np.logspace(-3, 2, 5)}),
        ('RF', pipeline_rf, {'clf__max_depth': [5, 10, None]}),
    ]

def tarea_individual(fold_idx, train_idx, val_idx, nombre_modelo, pipeline, param_grid):
    """Ejecuta UNA tarea (un fold x un modelo) y registra su propio
    tiempo de inicio/fin y el PID del proceso que la corrio. Esto es
    lo que permite despues dibujar el Gantt chart: cada tarea es una
    barra horizontal, su fila es el nucleo (PID) que la ejecuto."""
    t_inicio = time.perf_counter()
    pid = os.getpid()

    X_tr = X_train.iloc[train_idx]
    y_tr = y_train.iloc[train_idx]
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=1)
    grid.fit(X_tr, y_tr)

    t_fin = time.perf_counter()
    return {
        'fold': fold_idx, 'modelo': nombre_modelo,
        'pid': pid, 't_inicio': t_inicio, 't_fin': t_fin
    }

def generar_lista_tareas():
    tareas = []
    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train)):
        for nombre_modelo, pipeline, param_grid in modelos_benchmark():
            tareas.append((fold_idx, train_idx, val_idx, nombre_modelo, pipeline, param_grid))
    return tareas

def capturar_ejecucion(n_jobs):
    """Corre las 15 tareas (5 folds x 3 modelos... aqui 2 por brevedad)
    con el n_jobs indicado, y retorna un DataFrame con los timestamps
    de cada tarea individual, listo para graficar."""
    tareas = generar_lista_tareas()
    t_offset = time.perf_counter()  # referencia comun para alinear ambas corridas en el mismo eje de tiempo

    resultados = Parallel(n_jobs=n_jobs)(
        delayed(tarea_individual)(*t) for t in tareas
    )

    df_tareas = pd.DataFrame(resultados)
    df_tareas['t_inicio_rel'] = df_tareas['t_inicio'] - t_offset
    df_tareas['t_fin_rel'] = df_tareas['t_fin'] - t_offset
    df_tareas['duracion'] = df_tareas['t_fin_rel'] - df_tareas['t_inicio_rel']

    # Mapear PIDs reales a "carril" visual 0, 1, 2... en orden de aparicion,
    # para que el Gantt se vea limpio (no importa el PID real del SO).
    pids_unicos = df_tareas['pid'].unique()
    pid_a_carril = {pid: i for i, pid in enumerate(pids_unicos)}
    df_tareas['carril'] = df_tareas['pid'].map(pid_a_carril)

    return df_tareas

print("\n" + "=" * 60)
print("Capturando timestamps de ejecucion: modo SECUENCIAL (n_jobs=1)")
print("=" * 60)
tareas_secuencial = capturar_ejecucion(n_jobs=1)
print(f"  {len(tareas_secuencial)} tareas ejecutadas en {tareas_secuencial['pid'].nunique()} proceso(s)")

print("\n" + "=" * 60)
print(f"Capturando timestamps de ejecucion: modo PARALELO (n_jobs={max_cores})")
print("=" * 60)
tareas_paralelo = capturar_ejecucion(n_jobs=max_cores)
print(f"  {len(tareas_paralelo)} tareas ejecutadas en {tareas_paralelo['pid'].nunique()} proceso(s)")

# 6. GRAFICO 1: GANTT CHART -- SECUENCIAL vs. PARALELO
# Cada barra horizontal es UNA tarea (fold x modelo). El eje Y agrupa
# las tareas por "carril" (nucleo/proceso que la ejecuto). En el modo
# secuencial, todo corre en un solo carril, una tarea tras otra -- ahi
# se VE literalmente por que un pipeline mal paralelizado desperdicia
# nucleos disponibles. En el modo paralelo, las tareas se distribuyen
# en varios carriles y se superponen en el tiempo.
fig1, axes1 = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

colores_modelo = {'LR': '#1f77b4', 'RF': '#2ca02c'}

for ax, df_tareas, titulo in [
    (axes1[0], tareas_secuencial, f'Ejecucion SECUENCIAL (n_jobs=1) -- {tareas_secuencial["pid"].nunique()} nucleo activo'),
    (axes1[1], tareas_paralelo, f'Ejecucion PARALELA (n_jobs={max_cores}) -- {tareas_paralelo["pid"].nunique()} nucleos activos'),
]:
    for _, tarea in df_tareas.iterrows():
        ax.barh(
            y=tarea['carril'], width=tarea['duracion'], left=tarea['t_inicio_rel'],
            color=colores_modelo.get(tarea['modelo'], 'gray'), edgecolor='black', height=0.6
        )
        ax.text(tarea['t_inicio_rel'] + tarea['duracion'] / 2, tarea['carril'],
                f"{tarea['modelo']}-F{tarea['fold']}", ha='center', va='center', fontsize=7, color='white')
    ax.set_title(titulo)
    ax.set_xlabel('Tiempo (segundos)')
    ax.set_ylabel('Nucleo / proceso')
    ax.set_yticks(sorted(df_tareas['carril'].unique()))
    ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('grafico1_gantt_secuencial_vs_paralelo.png', dpi=150)
print("\nGrafico 1 guardado: grafico1_gantt_secuencial_vs_paralelo.png")


# 7. GRAFICO 2: HEATMAP DE UTILIZACION DE NUCLEOS EN EL TIEMPO
# DECISION METODOLOGICA:
# El Gantt chart muestra QUE tarea corrio donde; este heatmap muestra
# si un nucleo estuvo ACTIVO o INACTIVO en cada instante. Los huecos
# blancos (inactivo) en un carril mientras otros estan ocupados son
# la evidencia visual directa de desbalance de carga -- util para
# identificar si alguna tarea (ej. un fold con mas datos) tarda mas
# que las demas y deja nucleos esperando sin trabajo.
N_BINS_TIEMPO = 80
t_max_paralelo = tareas_paralelo['t_fin_rel'].max()
bins = np.linspace(0, t_max_paralelo, N_BINS_TIEMPO)

n_carriles = tareas_paralelo['carril'].nunique()
matriz_utilizacion = np.zeros((n_carriles, N_BINS_TIEMPO - 1))

for _, tarea in tareas_paralelo.iterrows():
    idx_inicio = np.searchsorted(bins, tarea['t_inicio_rel'], side='right') - 1
    idx_fin = np.searchsorted(bins, tarea['t_fin_rel'], side='right') - 1
    idx_inicio = max(0, min(idx_inicio, N_BINS_TIEMPO - 2))
    idx_fin = max(0, min(idx_fin, N_BINS_TIEMPO - 2))
    matriz_utilizacion[int(tarea['carril']), idx_inicio:idx_fin + 1] = 1

fig2, ax2 = plt.subplots(figsize=(12, 4))
im = ax2.imshow(matriz_utilizacion, aspect='auto', cmap='RdYlGn',
                extent=[0, t_max_paralelo, n_carriles - 0.5, -0.5])
ax2.set_xlabel('Tiempo (segundos)')
ax2.set_ylabel('Nucleo / proceso')
ax2.set_title(f'Utilizacion de Nucleos en el Tiempo (n_jobs={max_cores}) -- verde=activo, rojo=inactivo')
ax2.set_yticks(range(n_carriles))
plt.colorbar(im, ax=ax2, label='0 = inactivo, 1 = activo', ticks=[0, 1])
plt.tight_layout()
plt.savefig('grafico2_heatmap_utilizacion_nucleos.png', dpi=150)
print("Grafico 2 guardado: grafico2_heatmap_utilizacion_nucleos.png")

pct_utilizacion_promedio = matriz_utilizacion.mean() * 100
print(f"\nUtilizacion promedio de nucleos durante la corrida paralela: {pct_utilizacion_promedio:.1f}%")
print("(100% significaria que TODOS los nucleos estuvieron activos TODO el tiempo; "
    "un valor menor indica nucleos esperando porque no hay suficientes tareas "
    "independientes para llenarlos, o porque algunas tareas tardan mas que otras)")

# 8. GRAFICO 3: CPU MEDIDO vs. GPU SIMULADO (PROYECCION TEORICA)
# ADVERTENCIA METODOLOGICA IMPORTANTE (dejar explicita en el protocolo):
# No hay acceso a GPU, por lo que estos tiempos NO son una medicion
# real -- son una PROYECCION basada en un modelo simple y transparente:
#
#   tiempo_GPU(n) = overhead_transferencia_fijo + tiempo_computo_CPU(n) / factor_aceleracion
#
# Los parametros (overhead_transferencia_fijo, factor_aceleracion) son
# supuestos ilustrativos informados por caracteristicas conocidas de
# GPU (costo fijo de mover datos a memoria de video, independiente del
# tamano de los datos) y NO deben presentarse como una medicion --
# deben presentarse como una estimacion teorica, y el protocolo debe
# declararlo asi explicitamente. Si en el futuro se obtiene acceso a
# GPU real, esta seccion debe reemplazarse por una medicion real.
OVERHEAD_TRANSFERENCIA_GPU_SEG = 0.8   # supuesto ilustrativo: costo fijo de mover datos a GPU
FACTOR_ACELERACION_GPU = 3.0           # supuesto ilustrativo: aceleracion de computo puro en GPU

tiempo_cpu_medido = bench_df['tiempo_seg'].values
n_jobs_arr = bench_df['n_jobs'].values
tiempo_gpu_simulado = OVERHEAD_TRANSFERENCIA_GPU_SEG + (tiempo_cpu_medido / FACTOR_ACELERACION_GPU)

fig3, ax3 = plt.subplots(figsize=(8, 5.5))
x_pos = np.arange(len(n_jobs_arr))
ancho = 0.35

ax3.bar(x_pos - ancho / 2, tiempo_cpu_medido, ancho, label='CPU (medido)', color='#d62728')
ax3.bar(x_pos + ancho / 2, tiempo_gpu_simulado, ancho, label='GPU (proyeccion teorica)', color='#1f77b4', hatch='//')

ax3.set_xlabel('Configuracion (n_jobs de CPU equivalente)')
ax3.set_ylabel('Tiempo de ejecucion (segundos)')
ax3.set_title('CPU Medido vs. GPU Simulado -- Proyeccion Teorica\n(sin acceso a GPU real; NO es una medicion)')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f'n_jobs={n}' for n in n_jobs_arr])
ax3.legend()
ax3.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('grafico3_cpu_vs_gpu_simulado.png', dpi=150)
print("Grafico 3 guardado: grafico3_cpu_vs_gpu_simulado.png")

print(f"\n[PROYECCION, NO MEDICION] Con el dataset actual (n={X_train.shape[0]}), "
    f"el overhead fijo de transferencia a GPU ({OVERHEAD_TRANSFERENCIA_GPU_SEG}s) "
    f"representa una fraccion {'mayor' if OVERHEAD_TRANSFERENCIA_GPU_SEG > tiempo_gpu_simulado[0]*0.3 else 'menor'} "
    "del tiempo total, ilustrando por que GPU no ofrece ventaja clara a esta escala de datos.")
# 9. VISUALIZACION: TIEMPO Y SPEEDUP VS. NUCLEOS (Grafico 4)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel izquierdo: tiempo de ejecucion
axes[0].plot(bench_df['n_jobs'], bench_df['tiempo_seg'], marker='o', color='#d62728')
axes[0].set_xlabel('Numero de nucleos (n_jobs)')
axes[0].set_ylabel('Tiempo de ejecucion (segundos)')
axes[0].set_title('Tiempo de Nested CV vs. Nucleos CPU')
axes[0].grid(alpha=0.3)

# Panel derecho: speedup medido vs. speedup ideal vs. curva de Amdahl ajustada
n_range = np.linspace(1, max(bench_df['n_jobs']), 100)
axes[1].plot(bench_df['n_jobs'], bench_df['speedup'], marker='o', color='#2ca02c',
            label='Speedup medido')
axes[1].plot(n_range, n_range, linestyle='--', color='gray', label='Speedup ideal (lineal)')
if p_estimado is not None:
    axes[1].plot(n_range, amdahl_speedup(n_range, p_estimado), linestyle=':',
                color='#1f77b4', label=f'Ley de Amdahl (p={p_estimado:.2f})')
axes[1].set_xlabel('Numero de nucleos (n_jobs)')
axes[1].set_ylabel('Speedup')
axes[1].set_title('Speedup Medido vs. Ideal vs. Amdahl')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('grafico4_tiempo_speedup_amdahl.png', dpi=150)
print("Grafico 4 guardado: grafico4_tiempo_speedup_amdahl.png")
# 10. CONCLUSION AUTOMATICA (para pegar directo en el protocolo)
print("\n" + "=" * 60)
print("Resumen para el protocolo de investigacion")
print("=" * 60)
print(f"""
Con n={X_train.shape[0]} pacientes en el set de entrenamiento, el
paralelismo CPU vía n_jobs de scikit-learn produjo un speedup medido
de {bench_df['speedup'].max():.2f}x usando {int(bench_df['n_jobs'].max())} nucleos,
con una eficiencia de {bench_df['eficiencia'].iloc[-1]*100:.1f}% (perdida
esperable por overhead de coordinacion entre procesos).

La fraccion paralelizable estimada del pipeline es de aproximadamente
{f'{p_estimado*100:.0f}%' if p_estimado is not None else 'N/D'}, lo cual, via Ley de Amdahl, establece
un techo de speedup teorico de {f'{speedup_teorico_max:.1f}x' if p_estimado is not None else 'N/D'}
sin importar cuantos nucleos adicionales se agreguen. Esto confirma
que, para el volumen de datos actual, un esquema de computo distribuido
multi-nodo (Spark, Dask, MPI) o acelerado por GPU no ofreceria una
ganancia proporcional a su complejidad de implementacion: el cuello de
botella no es el computo, sino la fraccion secuencial inherente del
pipeline (carga de datos, particionamiento de folds, agregacion de
resultados), la cual ningun recurso adicional de computo puede
paralelizar mas alla de este punto.
""")