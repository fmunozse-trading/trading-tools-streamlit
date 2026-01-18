# Usar una imagen base de Python oficial
# El proyecto requiere Python >= 3.13 según pyproject.toml
FROM python:3.13-slim

# Instalar uv copiándolo desde su imagen oficial (método recomendado)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de definición de dependencias
COPY pyproject.toml uv.lock ./

# Instalar las dependencias del proyecto
# --frozen: Asegura que se instalen exactamente las versiones del uv.lock
# --no-install-project: No instala el paquete actual, solo dependencias (optimización para Docker)
RUN uv sync --frozen --no-install-project

# Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto por defecto de Streamlit
EXPOSE 8501

# Comprobación de salud (Healthcheck)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando para iniciar la aplicación
# uv run ejecutará el comando dentro del entorno virtual creado por uv sync
CMD ["uv", "run", "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
