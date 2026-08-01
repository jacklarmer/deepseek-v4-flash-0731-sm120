#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731 control helper.
#   ./ds4-ctl.sh up | down | health | logs [n] | dry-run | gates | bench [maxtok] [conc] [runs]
#
# Reads the endpoint API key from a mode-0600 file so the secret never appears
# in shell history, process arguments, or this repository.
set -uo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

[ -f "$root/.env" ] || { echo "missing .env (copy .env.example)" >&2; exit 1; }
set -a; source "$root/.env"; set +a

if [ -z "${VLLM_API_KEY:-}" ]; then
  key_file="${INFERENCE_KEY_FILE:-$HOME/.config/inference-api-key}"
  if [ -r "$key_file" ]; then
    IFS= read -r VLLM_API_KEY < "$key_file"
    export VLLM_API_KEY
  fi
fi
: "${VLLM_API_KEY:?set VLLM_API_KEY or provide INFERENCE_KEY_FILE}"
: "${MODEL_ROOT:?set MODEL_ROOT in .env}"
PORT="${PORT:-8000}"
MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-0731}"

compose() { docker compose --project-directory "$root" -f "$root/docker-compose.yml" "$@"; }

case "${1:-}" in
  up)
    [ -d "$MODEL_ROOT" ] || { echo "MODEL_ROOT not a directory: $MODEL_ROOT" >&2; exit 1; }
    compose config --quiet && compose up -d
    echo "starting; poll with: $0 health"
    ;;

  down) compose down ;;

  logs) compose logs --tail "${2:-80}" ;;

  dry-run)
    # Print the exact vllm serve command the in-image launcher will build.
    docker run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
      -e DRY_RUN=1 \
      -e CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}" -e PORT="$PORT" \
      -e MODE="${MODE:-dspark}" -e BACKEND="${BACKEND:-b12x-a8}" \
      -e ALLREDUCE_MODE="${ALLREDUCE_MODE:-b12x}" \
      -e TP_SIZE="${TP_SIZE:-4}" -e DCP_SIZE=1 \
      -e DSPARK_TOKENS="${DSPARK_TOKENS:-5}" \
      -e DSPARK_DEPTH_MODE="${DSPARK_DEPTH_MODE:-fixed}" \
      -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}" \
      -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}" \
      -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}" \
      -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}" \
      -e LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}" \
      -e DSPARK_MODEL=/model \
      -v "$MODEL_ROOT":/model:ro "$IMAGE"
    ;;

  health)
    base="http://127.0.0.1:${PORT}"
    h=$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 2 --max-time 5 "$base/health" 2>/dev/null || true)
    m=$(curl -sS -o /tmp/ds4-models.json -w '%{http_code}' --connect-timeout 2 --max-time 10 \
        -H "Authorization: Bearer $VLLM_API_KEY" "$base/v1/models" 2>/dev/null || true)
    echo "health=$h models=$m"
    [ "$m" = "200" ] && python3 -c "
import json; d=json.load(open('/tmp/ds4-models.json'))['data'][0]
print('model=%s max_model_len=%s' % (d['id'], d.get('max_model_len')))"
    ;;

  gates) exec python3 "$root/bench/gates.py" "$PORT" "$MODEL_NAME" ;;

  bench) exec python3 "$root/bench/codebench.py" "$PORT" "$MODEL_NAME" "${2:-512}" "${3:-1}" "${4:-5}" ;;

  *) echo "usage: $0 {up|down|health|logs|dry-run|gates|bench}" >&2; exit 2 ;;
esac
