# Derivación matemática del controlador propuesto

> Documento de referencia para la sección de metodología del paper.
> Contiene la derivación completa de las Contribuciones C1 y C2.

---

## 1. IBVS clásico (baseline)

Sea **s** ∈ ℝ^m el vector de features en imagen y **s\*** el objetivo deseado.
El error imagen es:

```
e = s - s*  ∈ ℝ^m
```

La relación entre la variación de features y la velocidad de la cámara **v** ∈ ℝ^6 es:

```
ṡ = L(s, Z) · v
```

donde **L** ∈ ℝ^{m×6} es la **matriz de interacción** (interaction matrix).

Para un punto imagen (u, v) con profundidad Z:

```
L_p = [ -1/Z    0    u/Z    uv      -(1+u²)   v  ]
      [  0    -1/Z   v/Z   1+v²     -uv       -u  ]
```

La ley de control IBVS clásica busca hacer ė = -λ·e:

```
v = -λ · L⁺ · e
```

donde L⁺ = (LᵀL)⁻¹Lᵀ es la pseudoinversa de Moore-Penrose.

Para un brazo robótico, la velocidad cartesiana se convierte a velocidades articulares mediante el Jacobiano cinemático **J_r** ∈ ℝ^{6×n}:

```
q̇ = J_r⁺ · v = J_r⁺ · (-λ · L⁺ · e)
```

**Limitación del baseline**: no considera el costo energético de las configuraciones articulares.

---

## 2. Contribución C2: Jacobiano visual adaptativo

En vez de usar L con profundidad asumida constante, definimos el **Jacobiano imagen-robot** compuesto:

```
J_a = L · J_r  ∈ ℝ^{m×n}
```

Este Jacobiano relaciona directamente velocidades articulares con variación de features:

```
ṡ = J_a · q̇
```

**J_a se estima online** sin calibración previa usando la regla de Broyden:

```
J_a(k+1) = J_a(k) + α · [Δq - J_a(k)·Δs] · Δsᵀ / (‖Δs‖² + ε)
```

donde:
- Δq = q(k) - q(k-1): cambio articular medido
- Δs = s(k) - s(k-1): cambio de features medido
- α ∈ (0,1]: tasa de aprendizaje
- ε > 0: regularización numérica

**Estimación de profundidad por flujo óptico** (sin red neuronal externa):

La divergencia del campo de flujo óptico θ = ∂u/∂x + ∂v/∂y satisface:

```
θ ≈ -v_z / Z
```

Por tanto:

```
Z_est = -v_z / θ  ≈  f · |v_z_proxy| / (|θ| + ε)
```

donde v_z_proxy se estima como la magnitud media del flujo entre frames consecutivos.

---

## 3. Contribución C1: Ley de control con penalización de torque

### 3.1 Motivación

En IBVS clásico, la ley q̇ = -λ·J_a⁺·e minimiza ‖e‖² sin considerar el torque necesario en cada junta. Configuraciones con alto torque implican mayor consumo energético:

```
P = Σᵢ |τᵢ| · |q̇ᵢ|   (potencia mecánica por junta)
```

### 3.2 Formulación propuesta

Definimos la **matriz de penalización de torque**:

```
W_τ = diag(|τ₁|/τ₁ᵐᵃˣ, ..., |τₙ|/τₙᵐᵃˣ) ∈ ℝ^{n×n}
```

Cada elemento W_τ[i,i] ∈ [0,1] indica qué tan cargada está la junta i relativa a su límite.

La **ley de control propuesta** es:

```
q̇ = -(λ_s · I + λ_τ · W_τ) · J_a⁺ · e
```

Expandiendo:

```
q̇ = -λ_s · J_a⁺ · e   ← término de convergencia imagen
  - λ_τ · W_τ · J_a⁺ · e   ← término de penalización energética
```

### 3.3 Análisis de estabilidad (Lyapunov)

Candidato de Lyapunov: V = ½ eᵀe > 0

Derivada:

```
V̇ = eᵀ · ė = eᵀ · J_a · q̇
   = eᵀ · J_a · [-(λ_s·I + λ_τ·W_τ) · J_a⁺ · e]
   = -eᵀ · J_a · (λ_s·I + λ_τ·W_τ) · J_a⁺ · e
```

Si J_a tiene rango completo (m ≤ n) y J_a · J_a⁺ ≈ I:

```
V̇ ≈ -eᵀ · (λ_s·I + λ_τ·W_τ) · e
   = -λ_s‖e‖² - λ_τ · eᵀ·W_τ·e
```

Como λ_s > 0, λ_τ ≥ 0, y W_τ es semidefinida positiva:

```
V̇ ≤ -λ_s · ‖e‖²  < 0  ∀ e ≠ 0
```

**Conclusión**: el sistema es **asintóticamente estable** para cualquier λ_τ ≥ 0, y el término de penalización de torque no degrada la estabilidad — solo redirige el esfuerzo de control hacia juntas menos cargadas.

### 3.4 Análisis del trade-off λ_s / λ_τ

| λ_τ | Comportamiento |
|-----|---------------|
| 0   | IBVS clásico. Convergencia rápida, sin ahorro energético |
| 0.1 | Balance óptimo (verificado experimentalmente) |
| 0.3 | Reducción energética máxima, convergencia ligeramente más lenta |
| > 0.5 | Posible oscilación si W_τ cambia rápidamente |

---

## 4. Métricas de evaluación

### 4.1 Error de seguimiento
```
RMSE_e = √(1/T · Σₜ ‖e(t)‖²)  [px]
```

### 4.2 Energía total consumida
```
E_total = Σₜ P(t) · Δt = Σₜ (Σᵢ |τᵢ(t)| · |q̇ᵢ(t)|) · Δt  [J]
```

### 4.3 Reducción energética respecto al baseline
```
ΔE% = (E_baseline - E_proposed) / E_baseline × 100
```

### 4.4 Velocidad de convergencia
```
T_conv = mín{t : ‖e(t)‖ < 5 px}  [iteraciones]
```

---

## 5. Parámetros de simulación

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| f (focal) | 554 px | FoV=55° a 640px (cámara Gazebo) |
| λ_s | 0.5 | Estándar en literatura IBVS |
| λ_τ | 0.1 | Óptimo según análisis de sensibilidad |
| α (Broyden) | 0.5 | Balance velocidad/estabilidad adaptación |
| ρ (Tikhonov) | 1e-4 | Regularización pseudoinversa |
| Δt | 1/30 s | 30 Hz loop de control |
| τ_max (UR5) | [150,150,150,28,28,28] Nm | Especificación fabricante |
