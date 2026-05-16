# Informe Técnico: Arquitectura, Motivación Física y Diseño del Modelo CODEX-b GNN

Este documento expone con rigor científico y técnico la arquitectura del sistema de veto para CODEX-b. Se centra en la justificación física de las decisiones de diseño y en la lógica subyacente al procesamiento de datos extraídos del subdetector VELO de LHCb.

---

## 1. Motivación Física y Construcción del Grafo

La construcción del grafo explota la topología del detector VELO y la cinemática de las trayectorias de las partículas. El objetivo es discriminar eventos a nivel de grafo, distinguiendo entre eventos de fondo (como muones penetrantes) y eventos de señal (partículas de vida larga o LLPs).

### 1.1. Topología del Grafo: Un Modelo de Propagación de Partículas
El grafo se diseña para capturar la coherencia espacial de las trazas. Dado que el volumen del VELO está libre del campo magnético principal, las partículas viajan en trayectorias esencialmente rectas. La topología de conexión se estructura así:

1.  **Aristas Intra-módulo (Clustering local)**:
    *   **Motivación**: Conecta hits dentro del mismo sensor ($r < 5$ mm). Esto permite que la red identifique depósitos de carga compartidos o electrones secundarios (rayos delta) como parte de una misma interacción local.
2.  **Aristas Inter-módulo (KNN hacia M±1)**:
    *   **Motivación**: Las trazas atraviesan los planos de silicio secuencialmente. Conectar un hit en la capa $k$ con sus $k$-vecinos más cercanos en $k+1$ preselecciona segmentos de traza congruentes con un bajo ángulo de dispersión.
3.  **Aristas de Salto (Skip-Edges, KNN hacia M±2)**:
    *   **Motivación**: Proporcionan robustez ante ineficiencias de los sensores o regiones muertas. Permiten que la información "salte" un plano, manteniendo la continuidad topológica de la traza a lo largo del detector.

### 1.2. Mapeo y Selección de Atributos (Features Reales)
Para que el modelo aprenda la física del detector, inyectamos una serie de variables cuidadosamente seleccionadas y normalizadas:

*   **Atributos del Nodo (Continuos)**:
    *   **Posición ($x, y, z$)**: Coordenadas espaciales normalizadas.
    *   **Geometría Cilíndrica ($r_T, \phi, \eta$)**: Fundamentales para romper simetrías. $r_T$ detecta el origen radial, $\phi$ la dirección acimutal y $\eta$ la pseudorapidez (ángulo respecto al haz).
    *   **Contexto de Hit (`n_pix`, `codex_angle`)**: El número de píxeles activados y el ángulo proyectivo hacia el volumen de CODEX-b.
    *   **Ocupación y Grado (`module_occupancy_norm`, `degree`)**: Indican la densidad de hits en el módulo y cuántas conexiones tiene el nodo, ayudando a distinguir trazas aisladas de regiones de alta multiplicidad.
*   **Atributos del Nodo (Categóricos)**:
    *   **`module`**: El identificador del sensor (0-51). Se procesa mediante un **Embedding** para que la red aprenda las particularidades geométricas y de aceptación de cada plano de silicio.
*   **Atributos de las Aristas (Geometric Edges)**:
    *   Vectores de 10 dimensiones que codifican la relación relativa entre dos hits: diferencias espaciales ($\Delta x, \Delta y, \Delta z$), distancia euclídea, deltas cilíndricas ($\Delta r_T, \Delta \phi$) y el vector dirección unitario ($u_x, u_y, u_z$).
*   **Atributos Globales (Contexto del Evento)**:
    *   **`nVtx_per_event`, `nClu_per_event`, `nTrk_per_event`**: Proporcionan una visión macroscópica de la complejidad del evento (pileup), permitiendo que la red ajuste su respuesta según el ruido ambiental.

---

## 2. Construcción del Modelo: Interaction Network (IN)

La arquitectura central utiliza una **Interaction Network (IN)** (V3), optimizada para asimilar restricciones geométricas mediante paso de mensajes.

### 2.1. Dinámica del Paso de Mensajes (Message Passing)
En la IN, las aristas no son simples enlaces, sino funciones activas:
*   **Edge MLP (Relational Model)**: Ingiere $[x_i, x_j, edge\_attr_{ij}]$ y produce un mensaje rico en información relacional. Aprende a suprimir conexiones físicamente imposibles y a reforzar aquellas que forman trayectorias coherentes.
*   **Node MLP (Object Model)**: Actualiza el estado del nodo basándose en su estado anterior y la suma de mensajes recibidos. Tras varias capas, el nodo "conoce" la trayectoria completa a la que pertenece.

### 2.2. Profundidad y Diseño de Capas
*   **Profundidad (5 capas)**: Permite una propagación de información de hasta 5 módulos de distancia. Esto es crítico para capturar la linealidad global de las trazas de señal que atraviesan gran parte del VELO.
*   **Estabilidad (Pre-LN y SiLU)**: Aplicamos **LayerNorm** antes del procesamiento y usamos la función de activación **SiLU**. Esto evita el desvanecimiento de gradientes y asegura que la red pueda aprender variaciones sutiles en los ángulos de las partículas.

### 2.3. Pooling y Clasificación (Graph Readout)
Dado que el objetivo es vetar el evento completo, se proyecta el grafo a un vector de decisión:
*   **Simplified Jumping Knowledge**: Extraemos características de la capa 3 (fragmentos de trazas) y la capa 5 (visión global).
*   **Pooling Híbrido**: Combinamos **Attentional Aggregation** (que asigna pesos de importancia a los hits) con **Global Max Pool** (que captura los hits más extremos o energéticos).
*   **Fusión Global**: Concatenamos el resultado del pooling con las variables globales (`nVtx`, `nClu`, `nTrk`) antes de pasar por el clasificador final (MLP de 3 capas con Dropout para evitar overfitting).

---

## 3. Justificación de Optimizaciones

1.  **Chunks e IterableDataset**: Los datos se cargan en streaming desde archivos segmentados. Esto permite entrenar con datasets masivos sin saturar la RAM de la CPU, manteniendo la GPU siempre alimentada de datos (prefetching).
2.  **Balanced Sampling (1:1)**: El pipeline de entrenamiento equilibra automáticamente la señal y el fondo mediante un ciclo infinito sobre los archivos de menor frecuencia. Esto permite usar un `pos_weight=1.0` y simplifica la convergencia.
3.  **BF16-Mixed Precision**: Los cálculos de distancias al cuadrado generan valores con gran rango dinámico. BF16 evita los overflows y NaNs que suelen ocurrir con el formato FP16 tradicional, manteniendo la eficiencia del hardware moderno.
4.  **Expansión Progresiva**: El modelo comienza entrenando con un subconjunto de datos para estabilizar los pesos iniciales y luego expande gradualmente su horizonte de aprendizaje, acelerando la convergencia en las etapas finales.

---
*Este diseño integra los principios de la física de partículas (linealidad y geometría del VELO) con las arquitecturas de grafos más potentes, resultando en un sistema de veto robusto, escalable y físicamente motivado.*