#!/bin/bash
# =============================================================================
# Script de Despliegue - Sistema de Detección de Enfermedades en Cultivos
# Despliegue en consola Linux (Docker o systemd)
# =============================================================================

set -e

echo "============================================================"
echo "  DESPLIEGUE - Sistema de Detección de Enfermedades CNN"
echo "============================================================"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# --- OPCIÓN A: Docker ---
deploy_docker() {
    echo ""
    echo ">>> OPCIÓN A: Despliegue con Docker"
    echo "============================================================"

    echo "[1/4] Construyendo imagen Docker..."
    docker build -t cnn-api .

    echo "[2/4] Deteniendo contenedor anterior (si existe)..."
    docker stop cnn-api 2>/dev/null || true
    docker rm cnn-api 2>/dev/null || true

    echo "[3/4] Ejecutando contenedor..."
    docker run -d \
        -p 5000:5000 \
        --name cnn-api \
        --restart unless-stopped \
        cnn-api

    echo "[4/4] Verificando estado..."
    sleep 5
    docker ps | grep cnn-api

    echo ""
    echo "=== Verificación de logs ==="
    docker logs cnn-api --tail 20

    echo ""
    echo "=== Health check ==="
    curl -s http://localhost:5000/health | python3 -m json.tool

    echo ""
    echo "============================================================"
    echo "  API desplegada con Docker en http://localhost:5000"
    echo "============================================================"
    echo ""
    echo "Comandos útiles:"
    echo "  docker ps                          # Ver contenedor"
    echo "  docker logs cnn-api -f             # Ver logs en vivo"
    echo "  docker restart cnn-api             # Reiniciar"
    echo "  docker stop cnn-api                # Detener"
    echo "  docker rm cnn-api                  # Eliminar"
    echo "  docker exec -it cnn-api bash       # Shell en contenedor"
}

# --- OPCIÓN B: systemd ---
deploy_systemd() {
    echo ""
    echo ">>> OPCIÓN B: Despliegue con systemd"
    echo "============================================================"

    echo "[1/5] Creando entorno virtual..."
    python3 -m venv venv
    source venv/bin/activate

    echo "[2/5] Instalando dependencias..."
    pip install --upgrade pip
    pip install -r requirements.txt

    echo "[3/5] Creando directorio de logs..."
    sudo mkdir -p /var/log/cnn-api
    sudo chown www-data:www-data /var/log/cnn-api

    echo "[4/5] Copiando servicio systemd..."
    sudo cp deploy/cnn-api.service /etc/systemd/system/
    sudo systemctl daemon-reload

    echo "[5/5] Habilitando e iniciando servicio..."
    sudo systemctl enable cnn-api
    sudo systemctl start cnn-api

    sleep 3
    echo ""
    echo "=== Estado del servicio ==="
    sudo systemctl status cnn-api --no-pager

    echo ""
    echo "=== Health check ==="
    curl -s http://localhost:5000/health | python3 -m json.tool

    echo ""
    echo "============================================================"
    echo "  API desplegada con systemd en http://localhost:5000"
    echo "============================================================"
    echo ""
    echo "Comandos útiles:"
    echo "  sudo systemctl status cnn-api      # Ver estado"
    echo "  sudo systemctl restart cnn-api     # Reiniciar"
    echo "  sudo systemctl stop cnn-api        # Detener"
    echo "  sudo journalctl -u cnn-api -f      # Ver logs en vivo"
    echo "  sudo systemctl disable cnn-api     # Deshabilitar"
}

# --- Menú ---
echo ""
echo "Seleccione método de despliegue:"
echo "  1) Docker (recomendado)"
echo "  2) systemd"
echo "  3) Ambos (punto extra)"
echo ""
read -p "Opción [1-3]: " choice

case $choice in
    1)
        deploy_docker
        ;;
    2)
        deploy_systemd
        ;;
    3)
        deploy_docker
        deploy_systemd
        ;;
    *)
        echo "Opción inválida. Usando Docker por defecto."
        deploy_docker
        ;;
esac

echo ""
echo "¡Despliegue completado!"
