# 📈 App Estudios Trading - Calculadora de Riesgo de Ruina

Aplicación interactiva desarrollada en **Streamlit** para analizar la viabilidad matemática de estrategias de fondeo en trading mediante simulaciones de **Montecarlo**.

Esta herramienta permite a los traders estimar su **Riesgo de Ruina** y calcular la **Esperanza Matemática (EV)** de sus intentos de evaluación (cuentas de fondeo), basándose en parámetros estadísticos personalizados.

## ✨ Características Principales

*   **Simulación de Montecarlo**: Ejecuta miles de escenarios (100 a 10,000) para proyectar posibles resultados.
*   **Cálculo de Riesgo de Ruina**: Determina la probabilidad porcentual de perder el capital destinado a evaluaciones antes de lograr un retiro.
*   **Esperanza Matemática (EV)**: Calcula la rentabilidad promedio por intento basada en tu tasa de éxito y ratio riesgo/beneficio.
*   **Visualización Interactiva**: Gráficos dinámicos con **Plotly** que muestran la evolución del bankroll en cada simulación.
*   **Interfaz Optimizada**: Diseño compacto y limpio con estilos personalizados.

## 🛠️ Instalación

Este proyecto utiliza [uv](https://github.com/astral-sh/uv) para la gestión de dependencias y entornos virtuales, garantizando una instalación rápida y reproducible.

1.  **Clonar el repositorio:**

    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd app-estudios-trading
    ```

2.  **Instalar dependencias:**

    Asegúrate de tener `uv` instalado. Si no, instálalo:
    ```bash
    # En Windows (PowerShell)
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    # En macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    Luego, sincroniza el entorno:
    ```bash
    uv sync
    ```

## 🚀 Uso

Para iniciar la aplicación, ejecuta el siguiente comando desde la raíz del proyecto:

```bash
uv run streamlit run main.py
```

Esto abrirá automáticamente la aplicación en tu navegador predeterminado (usualmente en `http://localhost:8501`).

### Navegación

*   **Home (`main.py`)**: Página de bienvenida.
*   **Calculadora Montecarlo (`pages/montecarlo.py`)**: Accede desde la barra lateral para utilizar la herramienta de simulación.

## 🧮 Conceptos Matemáticos

La aplicación utiliza las siguientes fórmulas clave:

*   **Esperanza Matemática (EV)**:
    $$ EV = (P_{ganar} \times (Retiro - Coste)) + (P_{perder} \times (-Coste)) $$

    Donde:
    *   $P_{ganar}$ = Tasa de éxito (%).
    *   $P_{perder}$ = Tasa de fallo (%).
    *   $Retiro$ = Retiro promedio esperado.
    *   $Coste$ = Coste de la prueba de evaluación.

## 📂 Estructura del Proyecto

```text
app-estudios-trading/
├── main.py              # Punto de entrada de la aplicación (Home)
├── pages/
│   └── montecarlo.py    # Lógica y UI de la simulación Montecarlo
├── utils.py             # Utilidades y estilos globales (CSS)
├── pyproject.toml       # Definición de dependencias y metadatos (uv)
├── uv.lock              # Archivo de bloqueo de versiones
└── README.md            # Documentación del proyecto
```

## 📦 Dependencias

Las principales librerías utilizadas son:
*   [Streamlit](https://streamlit.io/): Framework para la interfaz web.
*   [Plotly](https://plotly.com/python/): Gráficos interactivos.
*   [Pandas](https://pandas.pydata.org/): Manipulación de datos.
*   [NumPy](https://numpy.org/): Cálculos numéricos.

Revisa `pyproject.toml` para ver las versiones específicas.

## 📄 Licencia

Este proyecto es para uso educativo y personal.
