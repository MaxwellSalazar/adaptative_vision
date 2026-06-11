# adaptive_visual_servo

> **Control visual servo adaptativo con optimización de torque articular para seguimiento de objetos en movimiento usando cámara monocular no calibrada**
>
> Simulación en ROS2 Humble + Gazebo · Python 3.10 · Target: *Robotics and Autonomous Systems* (Q1, Elsevier)

---

## Índice

1. [Descripción del proyecto](#1-descripción-del-proyecto)
2. [Contribuciones científicas](#2-contribuciones-científicas)
3. [Arquitectura del sistema](#3-arquitectura-del-sistema)
4. [Requisitos del sistema](#4-requisitos-del-sistema)
5. [Instalación paso a paso](#5-instalación-paso-a-paso)
6. [Estructura del repositorio](#6-estructura-del-repositorio)
7. [Uso: lanzar la simulación](#7-uso-lanzar-la-simulación)
8. [Experimentos y métricas](#8-experimentos-y-métricas)
9. [Análisis de resultados](#9-análisis-de-resultados)
10. [Tests unitarios](#10-tests-unitarios)
11. [Descripción de módulos](#11-descripción-de-módulos)
12. [Parámetros configurables](#12-parámetros-configurables)
13. [Protocolo del benchmark completo](#13-protocolo-del-benchmark-completo)
14. [Referencias matemáticas](#14-referencias-matemáticas)
15. [Publicación objetivo](#15-publicación-objetivo)

---

## 1. Descripción del proyecto

Este repositorio implementa un sistema de **control visual servo (IBVS) adaptativo** para el seguimiento de objetos en movimiento con un brazo robótico UR5, usando exclusivamente una cámara monocular no calibrada montada en el efector final (*eye-in-hand*).

El aporte central respecto al estado del arte es la integración del **costo energético articular** directamente en la ley de control IBVS, mediante una matriz de penalización de torque `W_τ` que pondera el esfuerzo de cada junta proporcionalmente a su carga actual. Esto permite reducir el consumo energético manteniendo la estabilidad y precisión de seguimiento.

Todo el sistema corre íntegramente en simulación (ROS2 Humble + Gazebo Harmonic), sin necesidad de hardware físico.

---

## 2. Contribuciones científicas

### C1 — Ley de control IBVS con penalización de torque articular

La ley de control estándar `q̇ = -λ·J⁺·e` se modifica a:

```
q̇ = -(λ_s·I + λ_τ·W_τ) · J_a⁺ · e
```

donde `W_τ = diag(|τᵢ|/τᵢᵐᵃˣ)` penaliza juntas con alta carga, redirigiendo el esfuerzo de control hacia configuraciones energéticamente eficientes.

**Por qué es novedoso**: la literatura existente optimiza trayectoria O control visual por separado. Esta formulación los unifica en una ley de control única, manteniendo la estabilidad asintótica demostrada por Lyapunov.

### C2 — Jacobiano visual adaptativo sin calibración + profundidad por flujo óptico

El Jacobiano imagen-robot `J_a` se estima *online* sin calibración previa usando la regla de Broyden:

```
J_a(k+1) = J_a(k) + α·[Δq - J_a(k)·Δs]·Δsᵀ / (‖Δs‖² + ε)
```

La profundidad Z se infiere de la divergencia del flujo óptico Lucas-Kanade, sin ninguna red neuronal externa (sin MiDaS, sin DPT).

### C3 — Benchmark reproducible con métricas energéticas estandarizadas

Pipeline completamente abierto en ROS2/Gazebo con:
- 3 escenarios de validación estandarizados (estático, lineal, sinusoidal)
- Métricas energéticas: `E = Σ |τᵢ|·|q̇ᵢ|·Δt` [Joules]
- CSV automático por experimento + figuras listas para publicación

---

## 3. Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Gazebo Harmonic                              │
│  ┌──────────┐    /camera/image_raw    ┌────────────────────────┐    │
│  │  UR5 +   │ ──────────────────────▶│   feature_tracker      │    │
│  │  Cámara  │                        │   (detección HSV +     │    │
│  │ monocular│                        │    4 puntos de interés)│    │
│  └────┬─────┘                        └──────────┬─────────────┘    │
│       │ /joint_states                            │ /vs/features_*   │
│       │ (q, q̇, τ)                               ▼                  │
│       │                              ┌────────────────────────┐    │
│       │                              │   depth_estimator      │    │
│       │                              │   (divergencia flujo   │    │
│       │                              │    óptico → Z_est)     │    │
│       │                              └──────────┬─────────────┘    │
│       │                                         │ /vs/depth_*      │
│       │          ┌──────────────────────────────▼──────────────┐   │
│       │          │         adaptive_vs_controller              │   │
│       │          │                                              │   │
│       │◀─────────│  q̇ = -(λ_s·I + λ_τ·W_τ) · J_a⁺ · e       │   │
│       │  /joint_ │                                              │   │
│       │  vel_cmd │  • Jacobiano adaptativo (Broyden online)    │   │
│       │          │  • Penalización de torque (C1)               │   │
│       │          │  • Estimación Z por flujo óptico (C2)        │   │
│       │          └──────────────────┬───────────────────────────┘   │
│       │                             │ /vs/metrics                   │
│       │                             ▼                               │
│       │                  ┌─────────────────────┐                   │
│       │                  │   energy_monitor    │──▶ CSV + JSON     │
│       │                  │   (registro E, τ, e)│    resultados     │
│       │                  └─────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Tópicos ROS2 principales

| Tópico | Tipo | Dirección | Descripción |
|--------|------|-----------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Gazebo → nodos | Frame de cámara 640×480 |
| `/joint_states` | `sensor_msgs/JointState` | Gazebo → controlador | q, q̇, τ del UR5 |
| `/vs/features_current` | `Float32MultiArray` | tracker → ctrl | Features actuales s [8 floats] |
| `/vs/features_desired` | `Float32MultiArray` | tracker → ctrl | Features deseadas s* [8 floats] |
| `/vs/depth_estimates` | `Float32MultiArray` | depth → ctrl | Z estimadas por punto [4 floats] |
| `/vs/target_detected` | `std_msgs/Bool` | tracker → ctrl | Objeto detectado o no |
| `/joint_group_vel_controller/commands` | `Float64MultiArray` | ctrl → Gazebo | q̇ comandada [6 floats, rad/s] |
| `/vs/metrics` | `Float32MultiArray` | ctrl → monitor | [err_px, τ_rms, P_W, effort, W_trace] |
| `/vs/energy_metrics` | `Float32MultiArray` | ctrl → monitor | [E_J, P_mean, P_peak, τ_rms, iters] |
| `/vs/feature_image` | `sensor_msgs/Image` | tracker → RViz | Imagen con features dibujadas |

---

## 4. Requisitos del sistema

### Software obligatorio

| Componente | Versión | Instalación |
|-----------|---------|-------------|
| Ubuntu | 22.04 LTS | — |
| ROS2 | Humble | [docs.ros.org](https://docs.ros.org/en/humble/Installation.html) |
| Gazebo | Harmonic (gz-sim 8) | `sudo apt install ros-humble-ros-gz` |
| Python | 3.10 | Incluido en Ubuntu 22.04 |
| colcon | latest | `sudo apt install python3-colcon-common-extensions` |

### Paquetes ROS2 adicionales

```bash
sudo apt install \
  ros-humble-ur-description \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-gz-ros2-control \
  ros-humble-cv-bridge \
  ros-humble-image-transport \
  ros-humble-tf2-ros
```

### Dependencias Python (en entorno virtual)

Listadas en `requirements.txt`:
- `opencv-python >= 4.8`
- `numpy >= 1.24`
- `scipy >= 1.11`
- `matplotlib >= 3.7`
- `pandas >= 2.0`
- `pytest >= 7.4`

---

## 5. Instalación paso a paso

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/<usuario>/adaptive_visual_servo.git
cd adaptive_visual_servo
```

### Paso 2: Crear el entorno virtual Python

```bash
bash setup_venv.sh
```

Esto crea `.venv/` en la raíz del proyecto e instala todas las dependencias.

**En VSCode**: abrir `adaptive_vs_ws.code-workspace` y seleccionar `.venv/bin/python` como intérprete (Ctrl+Shift+P → *Python: Select Interpreter*).

### Paso 3: Compilar el paquete ROS2

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select adaptive_visual_servo
source install/setup.bash
```

### Paso 4: Verificar instalación

```bash
# Activar el entorno virtual
source .venv/bin/activate

# Ejecutar tests unitarios (sin ROS2)
bash run_tests.sh
```

Todos los tests deben pasar. Si alguno falla, revisar la sección de [Troubleshooting](#troubleshooting).

---

## 6. Estructura del repositorio

```
adaptive_vs_ws/
│
├── src/adaptive_visual_servo/          # Paquete ROS2
│   │
│   ├── adaptive_visual_servo/          # Módulos Python
│   │   ├── __init__.py
│   │   ├── adaptive_jacobian.py        # ★ CORE: C1 + C2 (sin dependencias ROS)
│   │   ├── adaptive_vs_controller.py   # Nodo ROS2: controlador propuesto
│   │   ├── ibvs_controller.py          # Nodo ROS2: baseline clásico
│   │   ├── feature_tracker.py          # Nodo ROS2: detección + tracking HSV
│   │   ├── depth_estimator.py          # Nodo ROS2: profundidad por flujo óptico
│   │   ├── energy_monitor.py           # Nodo ROS2: registro CSV de métricas
│   │   └── target_publisher.py         # Nodo ROS2: generador de escenarios
│   │
│   ├── launch/
│   │   ├── simulation.launch.py        # Launch principal (un experimento)
│   │   └── benchmark.launch.py         # Protocolo benchmark completo
│   │
│   ├── config/
│   │   ├── controller.yaml             # λ_s, λ_τ, tau_max, vel_limit
│   │   ├── tracker.yaml                # Rangos HSV, tamaño deseado
│   │   └── depth.yaml                  # Focal length, parámetros LK
│   │
│   ├── worlds/
│   │   └── vs_arena.world              # Mesa + esfera naranja + iluminación
│   │
│   ├── urdf/
│   │   └── ur5_camera.urdf             # UR5 + cámara monocular eye-in-hand
│   │
│   ├── rviz/
│   │   └── vs_system.rviz              # Configuración de visualización
│   │
│   ├── package.xml
│   └── setup.py
│
├── scripts/
│   ├── analyze_results.py              # Genera Figs 1-3 + Tabla 1 del paper
│   ├── sensitivity_analysis.py         # Genera Fig 4: análisis λ_τ
│   └── plot_realtime.py                # Monitor en tiempo real (con/sin ROS2)
│
├── tests/
│   ├── test_adaptive_jacobian.py       # Tests unitarios del módulo core
│   └── test_simulation_mock.py         # Tests de integración (sin ROS2)
│
├── docs/
│   └── math_derivation.md              # Derivación matemática completa
│
├── adaptive_vs_ws.code-workspace       # Workspace VSCode preconfigurado
├── requirements.txt                    # Dependencias Python del venv
├── setup_venv.sh                       # Script de creación del venv
├── run_tests.sh                        # Script de tests + análisis de sensibilidad
├── colcon.meta                         # Configuración de build colcon
├── .gitignore
└── README.md
```

**Archivo más importante**: `adaptive_visual_servo/adaptive_jacobian.py` — contiene las contribuciones C1 y C2 puras, sin dependencias ROS2. Se puede importar y testear directamente desde Python.

---

## 7. Uso: lanzar la simulación

### Setup previo (en cada terminal nueva)

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
source .venv/bin/activate
```

### Experimento estándar: método propuesto

```bash
ros2 launch adaptive_visual_servo simulation.launch.py \
    scenario:=static \
    mode:=proposed
```

### Experimento baseline (IBVS clásico)

```bash
ros2 launch adaptive_visual_servo simulation.launch.py \
    scenario:=static \
    mode:=baseline
```

### Los 3 escenarios disponibles

```bash
# Escenario 1: objeto estático
scenario:=static

# Escenario 2: movimiento lineal (senoidal en eje Y)
scenario:=linear

# Escenario 3: trayectoria sinusoidal Lissajous en plano XY
scenario:=sinusoidal
```

### Ajustar parámetros de control

```bash
ros2 launch adaptive_visual_servo simulation.launch.py \
    scenario:=linear \
    mode:=proposed \
    lambda_s:=0.5 \
    lambda_tau:=0.2 \
    gui:=true
```

### Monitor en tiempo real (terminal separado)

```bash
# Con ROS2 activo:
python scripts/plot_realtime.py

# Sin ROS2 (modo demo):
python scripts/plot_realtime.py
```

### Visualización en RViz2

```bash
ros2 run rviz2 rviz2 -d src/adaptive_visual_servo/rviz/vs_system.rviz
```

---

## 8. Experimentos y métricas

### Los 6 experimentos del paper

| # | Escenario | Modo | Archivo CSV generado |
|---|-----------|------|---------------------|
| 1 | static | baseline | `metrics_static_baseline_<ts>.csv` |
| 2 | static | proposed | `metrics_static_proposed_<ts>.csv` |
| 3 | linear | baseline | `metrics_linear_baseline_<ts>.csv` |
| 4 | linear | proposed | `metrics_linear_proposed_<ts>.csv` |
| 5 | sinusoidal | baseline | `metrics_sinusoidal_baseline_<ts>.csv` |
| 6 | sinusoidal | proposed | `metrics_sinusoidal_proposed_<ts>.csv` |

Los CSVs se guardan automáticamente en `~/vs_results/` al finalizar cada experimento.

### Columnas del CSV

| Columna | Unidad | Descripción |
|---------|--------|-------------|
| `timestamp_s` | s | Tiempo desde inicio del experimento |
| `image_error_norm_px` | px | ‖e‖₂ = ‖s - s*‖ en espacio imagen |
| `joint_torque_rms_Nm` | Nm | √(mean(τᵢ²)) — torque RMS global |
| `instantaneous_power_W` | W | P = Σᵢ\|τᵢ\|·\|q̇ᵢ\| — potencia mecánica |
| `control_effort_rad_s` | rad/s | ‖q̇‖₂ — magnitud del comando |
| `torque_penalty_trace` | — | tr(W_τ) — indicador de carga articular |

### Métricas objetivo para publicación

| Métrica | Objetivo | Descripción |
|---------|----------|-------------|
| Reducción energética | ≥ 15% | `(E_baseline - E_proposed) / E_baseline × 100` |
| Error steady-state | < 5 px | Error en espacio imagen tras convergencia |
| Latencia loop control | < 33 ms | Período del loop a 30 Hz |
| Degradación convergencia | < 30% | Iteraciones adicionales vs baseline |

---

## 9. Análisis de resultados

### Generar figuras del paper (Figs 1–3 + Tabla 1)

```bash
source .venv/bin/activate

python scripts/analyze_results.py \
    --baseline  "~/vs_results/metrics_static_baseline_*.csv" \
    --proposed  "~/vs_results/metrics_static_proposed_*.csv" \
    --output    "~/vs_results/figures/"
```

**Figuras generadas**:
- `fig1_convergence.pdf` — Curvas de convergencia del error imagen
- `fig2_energy.pdf` — Potencia instantánea y energía acumulada
- `fig3_torque_rms.pdf` — Torque RMS comparativo
- Tabla 1 impresa en consola (lista para copiar al paper)

### Análisis de sensibilidad de λ_τ (Fig 4)

```bash
python scripts/sensitivity_analysis.py
```

No requiere ROS2 — simula el controlador en lazo cerrado en Python puro.

**Figuras generadas** en `~/vs_results/figures/`:
- `fig4_sensitivity.pdf` — 3 subplots: precisión, energía, convergencia vs λ_τ
- `sensitivity_data.csv` — datos numéricos del barrido

---

## 10. Tests unitarios

```bash
source .venv/bin/activate
bash run_tests.sh
```

O directamente con pytest:

```bash
pytest tests/ -v --tb=short
```

### Cobertura de tests

| Módulo | Tests | Qué verifica |
|--------|-------|-------------|
| `test_adaptive_jacobian.py` | 10 tests | Shapes, convergencia Broyden, pseudoinversa, límites de velocidad, equivalencia con baseline cuando λ_τ=0 |
| `test_simulation_mock.py` | 9 tests | Convergencia en 3 escenarios, reducción energética ≥10%, robustez ante ruido y Jacobiano singular |

**En VSCode**: los tests son detectados automáticamente (Testing panel lateral).

---

## 11. Descripción de módulos

### `adaptive_jacobian.py` — Módulo core (sin ROS2)

El módulo más importante del proyecto. Contiene las contribuciones C1 y C2 en Python puro, importable sin ROS2.

```python
from adaptive_visual_servo.adaptive_jacobian import TorquePenalizedController

ctrl = TorquePenalizedController(
    n_joints=6,
    lambda_s=0.5,    # ganancia imagen
    lambda_tau=0.1,  # peso energético (0 = IBVS clásico)
)

# En cada paso del loop de control:
q_dot, metrics = ctrl.compute_control(
    image_error=e,        # s - s* [8-vector]
    tau_current=tau,      # torques actuales [6-vector, Nm]
)

# Actualizar Jacobiano con observación real:
ctrl.update_jacobian(delta_q, delta_s)

# Resumen energético:
summary = ctrl.get_energy_summary()
# → {'total_energy_joules': 42.3, 'mean_power_watts': 8.1, ...}
```

**Clases exportadas**:
- `TorquePenalizedController` — controlador principal (C1+C2)
- `AdaptiveVisualJacobian` — Jacobiano adaptativo Broyden (C2)
- `ImageInteractionMatrix` — matriz de interacción L(s,Z) (IBVS)

### `feature_tracker.py` — Detección y tracking de features

Detecta la esfera naranja por segmentación HSV y extrae 4 puntos del bounding box como features imagen. Publica:
- `/vs/features_current` — coordenadas s = [u1,v1,...,u4,v4]
- `/vs/feature_image` — imagen con overlay para RViz

Ajustar los rangos HSV en `config/tracker.yaml` si el objeto cambia de color.

### `depth_estimator.py` — Estimación de profundidad (C2)

Estima Z de los puntos de interés usando divergencia de flujo óptico Lucas-Kanade. No usa ninguna red neuronal; el cálculo es analítico y verificable.

### `energy_monitor.py` — Registro de métricas

Escribe un CSV por experimento con todas las métricas en tiempo real. Se cierra correctamente al interrumpir con Ctrl+C.

### `target_publisher.py` — Generador de escenarios

Controla la posición de la esfera en Gazebo según el escenario activo. Los perfiles de movimiento están en el diccionario `SCENARIOS` de la clase.

---

## 12. Parámetros configurables

### `config/controller.yaml`

```yaml
adaptive_vs_controller:
  ros__parameters:
    lambda_s: 0.5        # Ganancia imagen [0.1–1.0]
    lambda_tau: 0.1      # Peso energético [0.0–0.5]
    joint_vel_limit: 0.8 # Límite vel. articular [rad/s]
    control_rate_hz: 30.0
    tau_max: [150.0, 150.0, 150.0, 28.0, 28.0, 28.0]  # UR5 [Nm]
```

**Cómo elegir λ_τ**: usar `scripts/sensitivity_analysis.py` para encontrar el valor óptimo. Típicamente λ_τ ∈ [0.05, 0.2] da el mejor balance precisión/energía.

### `config/tracker.yaml`

```yaml
feature_tracker:
  ros__parameters:
    hsv_lower: [10, 100, 100]  # Ajustar si el objeto tiene otro color
    hsv_upper: [30, 255, 255]
    desired_u: 320.0           # Centro horizontal imagen
    desired_v: 240.0           # Centro vertical imagen
    desired_size_px: 80.0      # Tamaño deseado del objeto en imagen
```

---

## 13. Protocolo del benchmark completo

Para reproducir todos los resultados del paper:

```bash
# Terminal 1: setup
source /opt/ros/humble/setup.bash && source install/setup.bash && source .venv/bin/activate

# Experimento 1/6
ros2 launch adaptive_visual_servo simulation.launch.py scenario:=static mode:=baseline
# Esperar 90 segundos, Ctrl+C

# Experimento 2/6
ros2 launch adaptive_visual_servo simulation.launch.py scenario:=static mode:=proposed
# Esperar 90 segundos, Ctrl+C

# Repetir para linear y sinusoidal...

# Análisis final
python scripts/analyze_results.py \
    --baseline "~/vs_results/metrics_static_baseline_*.csv" \
    --proposed "~/vs_results/metrics_static_proposed_*.csv" \
    --output   "~/vs_results/figures/"

python scripts/sensitivity_analysis.py
```

Los CSVs de todos los experimentos quedan en `~/vs_results/` con timestamp para identificarlos.

---

## 14. Referencias matemáticas

Ver `docs/math_derivation.md` para la derivación completa que incluye:
- Matriz de interacción L(s, Z) para N puntos
- Demostración de estabilidad Lyapunov del controlador propuesto
- Análisis del trade-off λ_s/λ_τ
- Estimación de profundidad por divergencia de flujo óptico
- Tabla de parámetros con justificación

---

## 15. Publicación objetivo

| Prioridad | Revista | Factor de impacto | Cuartil |
|-----------|---------|-------------------|---------|
| 1ª | Robotics and Autonomous Systems (Elsevier) | ~4.3 | Q1 |
| 2ª | IEEE/ASME Transactions on Mechatronics | ~5.8 | Q1 |
| 3ª | Journal of Intelligent & Robotic Systems (Springer) | ~3.3 | Q2 |
| 4ª | Mechatronics (Elsevier) | ~3.7 | Q2 |

**Ventaja de RAS como primera opción**: acepta explícitamente trabajos con validación en simulación siempre que el marco teórico sea sólido y el benchmark sea reproducible. El pipeline abierto en ROS2/Gazebo y la disponibilidad del código en GitHub cumplen ese criterio.

---

## Troubleshooting

**Error: `ModuleNotFoundError: adaptive_visual_servo`**
```bash
pip install -e src/adaptive_visual_servo
```

**Error: Gazebo no arranca**
```bash
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$(pwd)/src/adaptive_visual_servo
```

**Error: `/joint_states` vacío (sin torques)**
Verificar que el plugin `gz-sim-joint-state-publisher` está activo en el world. El controlador tiene fallback que estima τ desde velocidades.

**Tests fallan por imports de ROS2**
Los tests están diseñados para correr sin ROS2 activo. Si hay un error de import, verificar que el entorno virtual está activo:
```bash
source .venv/bin/activate
which python  # debe apuntar a .venv/bin/python
```

---

## Licencia

MIT License — ver [LICENSE](LICENSE)

---

*Desarrollado para investigación doctoral. Pipeline reproducible con código abierto.*
