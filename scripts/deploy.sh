#!/usr/bin/env bash
# Deploy the F1 Strategy stack to the production server.
# Run from the repo root: bash scripts/deploy.sh
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[deploy]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
error() { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ── 1. Preflight checks ───────────────────────────────────────────────────────
info "Preflight checks..."

[[ -f "docker-compose.yml" ]]      || error "Run from repo root (docker-compose.yml not found)"
[[ -f "docker-compose.prod.yml" ]] || error "docker-compose.prod.yml not found"

command -v docker   >/dev/null || error "docker is not installed"
command -v openssl  >/dev/null || warn  "openssl not found — skip cert generation if certs already exist"

# ── 2. TLS certificates ───────────────────────────────────────────────────────
CERT_DIR="monitoring/nginx/certs"
CERT_FILE="${CERT_DIR}/server.crt"
KEY_FILE="${CERT_DIR}/server.key"
mkdir -p "${CERT_DIR}"

if [[ -f "${CERT_FILE}" && -f "${KEY_FILE}" ]]; then
    info "TLS certs already present — skipping generation."
else
    info "Generating self-signed TLS certificate..."
    command -v openssl >/dev/null || error "openssl required to generate certs"
    SERVER_IP="${F1_SERVER_IP:-167.233.125.215}"
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "${KEY_FILE}" \
        -out    "${CERT_FILE}" \
        -subj   "/CN=${SERVER_IP}" \
        -addext "subjectAltName=IP:${SERVER_IP}" 2>/dev/null
    chmod 600 "${KEY_FILE}"
    info "Certificate written to ${CERT_FILE}"
fi

# ── 3. Validate .env.production ───────────────────────────────────────────────
ENV_FILE=".env.production"
[[ -f "${ENV_FILE}" ]] || error "${ENV_FILE} not found. Copy .env.production.example and fill in secrets."

# Check for unset placeholders
if grep -q "CHANGE_ME" "${ENV_FILE}"; then
    error "${ENV_FILE} still contains CHANGE_ME placeholders. Set real values before deploying."
fi

GF_PASS=$(grep "^GF_ADMIN_PASSWORD=" "${ENV_FILE}" | cut -d= -f2- | tr -d '"' || true)
if [[ -z "${GF_PASS}" ]]; then
    error "GF_ADMIN_PASSWORD is not set in ${ENV_FILE}"
fi
if [[ "${GF_PASS}" == "admin" ]]; then
    warn "GF_ADMIN_PASSWORD is 'admin' — change it before exposing to the internet."
fi

info ".env.production validated."

# ── 4. Build and start ────────────────────────────────────────────────────────
COMPOSE_CMD="docker compose -f docker-compose.yml -f docker-compose.prod.yml"

info "Pulling latest images..."
${COMPOSE_CMD} pull --ignore-pull-failures 2>/dev/null || true

info "Building API image..."
${COMPOSE_CMD} build \
    --build-arg F1_BUILD_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
    --build-arg F1_BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

info "Starting services..."
${COMPOSE_CMD} up -d --remove-orphans

# ── 5. Health check ───────────────────────────────────────────────────────────
info "Waiting for API health check..."
RETRIES=30
for i in $(seq 1 ${RETRIES}); do
    if ${COMPOSE_CMD} exec -T api \
        python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
        >/dev/null 2>&1; then
        info "API is healthy."
        break
    fi
    if [[ "${i}" == "${RETRIES}" ]]; then
        error "API failed health check after ${RETRIES} attempts. Check: docker compose logs api"
    fi
    sleep 2
done

# ── 6. Summary ────────────────────────────────────────────────────────────────
SERVER_IP="${F1_SERVER_IP:-167.233.125.215}"
echo ""
info "Deploy complete."
echo -e "  API:       ${GREEN}https://${SERVER_IP}/${NC}"
echo -e "  Grafana:   ${GREEN}https://${SERVER_IP}/grafana/${NC}"
echo -e "  MLflow:    ${GREEN}https://${SERVER_IP}/mlflow/${NC}"
echo -e "  Alertmgr:  http://${SERVER_IP}:9093  (internal only)"
echo ""
warn "If this is the first deploy, run: make train-evaluate MODEL_BACKEND=xgboost"
warn "Then: make promote-artifact ARTIFACT_ID=<id from artifacts/models/registry.json>"
