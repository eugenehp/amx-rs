#!/usr/bin/env bash
#
# bench.sh — Run AMX hardware benchmarks and persist results per chip.
#
# Usage:
#   ./bench.sh              # run and save
#   ./bench.sh --open       # run, save, and open the result
#   ./bench.sh --list       # list all saved results
#   ./bench.sh --compare    # side-by-side of all saved chips
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$ROOT_DIR/benchmark-results"
mkdir -p "$RESULTS_DIR"

# ── Colors ──

if [[ -t 1 ]]; then
  BOLD='\033[1m'
  DIM='\033[2m'
  GREEN='\033[32m'
  CYAN='\033[36m'
  YELLOW='\033[33m'
  RED='\033[31m'
  RESET='\033[0m'
else
  BOLD='' DIM='' GREEN='' CYAN='' YELLOW='' RED='' RESET=''
fi

info()  { echo -e "${CYAN}▸${RESET} $*"; }
ok()    { echo -e "${GREEN}✓${RESET} $*"; }
warn()  { echo -e "${YELLOW}⚠${RESET} $*"; }
err()   { echo -e "${RED}✗${RESET} $*" >&2; }
step()  { echo -e "\n${BOLD}── $* ──${RESET}"; }

# ── Args ──

ACTION="run"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --open)    ACTION="run-open" ;;
    --list)    ACTION="list" ;;
    --compare) ACTION="compare" ;;
    -h|--help)
      echo "Usage: ./bench.sh [--open] [--list] [--compare]"
      echo "  (default)   Run benchmark, save to benchmark-results/"
      echo "  --open      Run + open result file"
      echo "  --list      List all saved results"
      echo "  --compare   Print side-by-side comparison of all chips"
      exit 0
      ;;
    *) err "Unknown arg: $1"; exit 1 ;;
  esac
  shift
done

# ── Helpers ──

detect_chip() {
  sysctl -n machdep.cpu.brand_string 2>/dev/null | sed 's/ *$//' || echo "unknown"
}

chip_slug() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-//;s/-$//'
}

timestamp() {
  date +%Y%m%d-%H%M%S
}

# ── List ──

if [[ "$ACTION" == "list" ]]; then
  step "Saved benchmark results"
  echo
  count=0
  for f in "$RESULTS_DIR"/*.txt; do
    [[ -f "$f" ]] || continue
    count=$((count + 1))
    chip="$(head -1 "$f" | sed 's/^# //')"
    date_line="$(sed -n '2p' "$f" | sed 's/^# //')"
    size="$(wc -c < "$f" | tr -d ' ')"
    printf "  ${BOLD}%-45s${RESET} ${DIM}%s${RESET}  ${DIM}%s${RESET}\n" \
      "$chip" "$date_line" "$(( size / 1024 ))KB"
  done
  [[ $count -eq 0 ]] && echo -e "  ${DIM}(none — run ./bench.sh first)${RESET}"
  echo
  info "$count result(s) in $RESULTS_DIR/"
  exit 0
fi

# ── Compare ──

if [[ "$ACTION" == "compare" ]]; then
  files=("$RESULTS_DIR"/*.txt)
  if [[ ! -f "${files[0]}" ]]; then
    err "No results found. Run ./bench.sh first."
    exit 1
  fi

  echo
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  AMX Cross-Chip Comparison                                      ║"
  echo "╚══════════════════════════════════════════════════════════════════╝"

  for f in "${files[@]}"; do
    [[ -f "$f" ]] || continue
    chip="$(head -1 "$f" | sed 's/^# //')"
    date_line="$(sed -n '2p' "$f" | sed 's/^# //')"
    echo
    echo -e "  ${BOLD}━━━ $chip ${DIM}($date_line)${RESET}"
    sed -n '/Single P-core/,/^$/p' "$f" | sed 's/^/  /'
    sed -n '/Whole chip/,/^$/p' "$f" | sed 's/^/  /'
  done
  echo
  exit 0
fi

# ── Run ──

CHIP="$(detect_chip)"
SLUG="$(chip_slug "$CHIP")"
TS="$(timestamp)"
OUTFILE="$RESULTS_DIR/${SLUG}-${TS}.txt"
RUST_VER="$(rustc --version 2>/dev/null || echo 'unknown')"
NCPU="$(sysctl -n hw.logicalcpu 2>/dev/null || echo '?')"
NPERF="$(sysctl -n hw.nperflevels 2>/dev/null || echo '1')"

# Detect if we need --target (x86_64 Rust under Rosetta on Apple Silicon)
TARGET_FLAG=""
TARGET_DIR="release"
HOST_ARCH="$(rustc -vV 2>/dev/null | grep '^host:' | awk '{print $2}')"
if [[ "$(uname -m)" == "arm64" && "$HOST_ARCH" != *aarch64* ]]; then
  TARGET_FLAG="--target aarch64-apple-darwin"
  TARGET_DIR="aarch64-apple-darwin/release"
  warn "Rust host is ${HOST_ARCH} (Rosetta?), cross-compiling with $TARGET_FLAG"
fi

step "System"
info "Chip:       ${BOLD}$CHIP${RESET}"
info "Cores:      $NCPU  (perf levels: $NPERF)"
info "OS:         $(uname -srm)"
info "Rust:       $RUST_VER  (host: $HOST_ARCH)"
info "Output:     $OUTFILE"

step "Build"
BUILD_START=$SECONDS
if ! cargo bench -p amx-sys --bench amx_bench $TARGET_FLAG --no-run 2>&1 | while IFS= read -r line; do
  echo -e "  ${DIM}$line${RESET}"
done; then
  err "Build failed. Full output:"
  cargo bench -p amx-sys --bench amx_bench $TARGET_FLAG --no-run 2>&1
  exit 1
fi
BUILD_SECS=$(( SECONDS - BUILD_START ))
ok "Built in ${BUILD_SECS}s"

# Find the bench binary
BENCH_BIN=""
for f in "$ROOT_DIR"/target/$TARGET_DIR/deps/amx_bench-*; do
  [[ -f "$f" && -x "$f" && "$f" != *.d && "$f" != *.rmeta ]] && BENCH_BIN="$f" && break
done

if [[ -z "$BENCH_BIN" ]]; then
  err "Cannot find amx_bench binary in target/$TARGET_DIR/deps/"
  err "Try: cargo bench -p amx-sys --bench amx_bench $TARGET_FLAG --no-run"
  exit 1
fi

step "Benchmark"
info "Binary: ${DIM}$(basename "$BENCH_BIN")${RESET}"
echo

BENCH_START=$SECONDS

# Run and tee to both stdout and file
{
  echo "# $CHIP"
  echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# $(uname -srm)"
  echo "# cores: $NCPU"
  echo "# perflevels: $NPERF"
  echo "# rust: $RUST_VER"
  echo
  "$BENCH_BIN" --bench 2>&1 || "$BENCH_BIN" 2>&1
} | tee "$OUTFILE"

BENCH_SECS=$(( SECONDS - BENCH_START ))

step "Done"
ok "Completed in ${BENCH_SECS}s"
ok "Saved to: ${BOLD}$OUTFILE${RESET}"
info "Run ${BOLD}./bench.sh --list${RESET} to see all results"
info "Run ${BOLD}./bench.sh --compare${RESET} for cross-chip comparison"

# ── Open ──

if [[ "$ACTION" == "run-open" ]]; then
  if command -v open >/dev/null 2>&1; then
    open "$OUTFILE"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$OUTFILE"
  else
    warn "Cannot open file automatically."
  fi
fi
