#!/usr/bin/env bash
# Colab GPU 런타임(또는 다른 CUDA Linux)에서 12개 sub-project의 triton_attn.py
# smoke test를 모두 실행하고 PASS/FAIL 집계.
#
# 사용법 (Colab 셀):
#   !git clone <repo-url>
#   %cd <repo>/vllm_attn
#   !bash colab_smoke_test.sh
#
# 또는 로컬 CUDA Linux:
#   cd vllm_attn && bash colab_smoke_test.sh
#
# 검증 범위: 각 sub-project의 `triton_attn.py`의 `if __name__ == "__main__":`
# smoke test (SDPA 대비 max_abs_err 측정). vLLM plugin 등록·E2E generate 는
# 별도 — `qwen3_triton_attention.ipynb` 사용.
#
# torch / triton 만 있으면 됨. vLLM 설치 불필요.
# Colab 기본 환경에 torch/triton 둘 다 사전 설치되어 있으므로 추가 설치도
# 보통 불필요.

# 의도: 어느 한 커널이 실패해도 나머지 다 돌려서 전체 그림 보기. set -e 안 씀.

# ---- 1. 환경 확인 ----------------------------------------------------------
echo "════════════════════════════════════════════════════════"
echo " vllm_attn smoke test (12 kernels)"
echo "════════════════════════════════════════════════════════"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "❌ nvidia-smi 없음. Colab은 [런타임 → 런타임 유형 변경 → GPU] 활성화 필요."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo

PYTHON="${PYTHON:-python3}"
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu128"

# torch는 CUDA 12.8 빌드로 고정. 기존 torch의 CUDA가 12.8이 아니면 재설치.
existing_cuda=$("$PYTHON" -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || echo "")
if [ "$existing_cuda" != "12.8" ]; then
    [ -n "$existing_cuda" ] && echo "ℹ 기존 torch CUDA=$existing_cuda — cu128로 재설치"
    "$PYTHON" -m pip install --quiet --index-url "$TORCH_CUDA_INDEX" torch || {
        echo "❌ torch (cu128) 설치 실패"; exit 1;
    }
fi
if ! "$PYTHON" -c "import triton" 2>/dev/null; then
    "$PYTHON" -m pip install --quiet triton || {
        echo "❌ triton 설치 실패"; exit 1;
    }
fi
"$PYTHON" -c "
import torch, triton
print(f'  torch  : {torch.__version__}  CUDA: {torch.version.cuda}  available: {torch.cuda.is_available()}')
print(f'  triton : {triton.__version__}')
assert torch.cuda.is_available(), 'CUDA 사용 불가 — GPU 런타임 활성화 필요'
assert torch.version.cuda == '12.8', f'torch CUDA 12.8 고정 — 실제: {torch.version.cuda}'
"
[ $? -ne 0 ] && exit 1
echo

# ---- 2. 검증 대상 12개 sub-project ----------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECTS=(
    "block_ptr/vllm_padded_decode"
    "block_ptr/vllm_split"
    "block_ptr/vllm_paged"
    "block_ptr/vllm_multiseq"
    "block_ptr/vllm_unified"
    "block_ptr/vllm_varlen"
    "ptr/vllm_padded_decode"
    "ptr/vllm_split"
    "ptr/vllm_paged"
    "ptr/vllm_multiseq"
    "ptr/vllm_unified"
    "ptr/vllm_varlen"
)

# ---- 3. 각각 실행 + 결과 집계 ----------------------------------------------
declare -a RESULTS
PASS_COUNT=0
FAIL_COUNT=0
LOG_DIR="$(mktemp -d -t vllm_attn_smoke_XXXX)"
echo "ℹ per-project log dir: $LOG_DIR"
echo

for p in "${PROJECTS[@]}"; do
    SCRIPT="$REPO_ROOT/$p/triton_attn.py"
    if [ ! -f "$SCRIPT" ]; then
        echo "⚠ SKIP $p (triton_attn.py 없음)"
        RESULTS+=("SKIP  $p")
        continue
    fi
    echo "──────────────────────────────────────────────────────"
    echo "▶  $p"
    echo "──────────────────────────────────────────────────────"
    LOG_FILE="$LOG_DIR/$(echo "$p" | tr '/' '_').log"

    # python을 명시적으로 분리해서 실행 — pipe 없이 정확한 exit code 캡처.
    # `|| true`로 한 번 더 가드: 어떤 이유든 다음 iteration으로 무조건 진행.
    "$PYTHON" "$SCRIPT" >"$LOG_FILE" 2>&1
    exit_code=$?
    cat "$LOG_FILE" || true

    LAST=$(tail -n 1 "$LOG_FILE" 2>/dev/null | tr -d '[:space:]')
    if [ "$exit_code" -eq 0 ] && [ "$LAST" = "ALLPASS" ]; then
        echo "✅ PASS  $p"
        RESULTS+=("PASS  $p")
        PASS_COUNT=$((PASS_COUNT+1))
    elif [ "$exit_code" -ne 0 ]; then
        # python이 traceback/assertion 등으로 죽었음 — 그래도 다음으로 진행
        echo "❌ FAIL  $p  (python exit code $exit_code) — 다음 sub-project로 진행"
        RESULTS+=("FAIL  $p  (python exit $exit_code)")
        FAIL_COUNT=$((FAIL_COUNT+1))
    else
        echo "⚠ FAIL  $p  (exit 0이지만 ALL PASS 아님: '$LAST') — 다음 sub-project로 진행"
        RESULTS+=("FAIL  $p  (last: $LAST)")
        FAIL_COUNT=$((FAIL_COUNT+1))
    fi
    echo
done

# ---- 4. 요약 ----------------------------------------------------------------
echo "════════════════════════════════════════════════════════"
echo " 요약"
echo "════════════════════════════════════════════════════════"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo "─────────────────────────────────────────"
echo "  TOTAL: ${#PROJECTS[@]}   PASS: $PASS_COUNT   FAIL: $FAIL_COUNT"
echo "  per-project logs: $LOG_DIR"
echo "════════════════════════════════════════════════════════"

[ $FAIL_COUNT -eq 0 ] && exit 0 || exit 1
