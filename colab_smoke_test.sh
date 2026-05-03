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
if ! "$PYTHON" -c "import torch, triton" 2>/dev/null; then
    echo "ℹ torch / triton 미설치 — pip로 설치 시도"
    "$PYTHON" -m pip install --quiet torch triton || {
        echo "❌ torch/triton 설치 실패"; exit 1;
    }
fi
"$PYTHON" -c "
import torch, triton
print(f'  torch  : {torch.__version__}  CUDA: {torch.version.cuda}  available: {torch.cuda.is_available()}')
print(f'  triton : {triton.__version__}')
assert torch.cuda.is_available(), 'CUDA 사용 불가 — GPU 런타임 활성화 필요'
"
[ $? -ne 0 ] && exit 1
echo

# ---- 2. 검증 대상 12개 sub-project ----------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECTS=(
    "ptr/vllm_padded_decode"
    "ptr/vllm_split"
    "ptr/vllm_paged"
    "ptr/vllm_multiseq"
    "ptr/vllm_unified"
    "ptr/vllm_varlen"
    "block_ptr/vllm_padded_decode"
    "block_ptr/vllm_split"
    "block_ptr/vllm_paged"
    "block_ptr/vllm_multiseq"
    "block_ptr/vllm_unified"
    "block_ptr/vllm_varlen"
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
    if "$PYTHON" "$SCRIPT" 2>&1 | tee "$LOG_FILE"; then
        # 마지막 줄이 ALL PASS 인지 확인 (smoke test 컨벤션)
        LAST=$(tail -n 1 "$LOG_FILE" | tr -d '[:space:]')
        if [ "$LAST" = "ALLPASS" ]; then
            echo "✅ PASS  $p"
            RESULTS+=("PASS  $p")
            PASS_COUNT=$((PASS_COUNT+1))
        else
            echo "⚠ EXIT 0 이지만 마지막 줄이 'ALL PASS' 아님: '$LAST' — FAIL 처리"
            RESULTS+=("FAIL  $p  (last line: $LAST)")
            FAIL_COUNT=$((FAIL_COUNT+1))
        fi
    else
        echo "❌ FAIL  $p  (exit code $?)"
        RESULTS+=("FAIL  $p  (non-zero exit)")
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
