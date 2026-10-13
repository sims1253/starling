#!/usr/bin/env bash
# Phone half of npu_compile.sh (run under the phone lock):
#   npu_compile_phone.sh NPU_TOOL TFLITE GGUF|0xHASH EXPECT_HASH KEY DIR APPS WORKDIR
# compiles TFLITE, a Qwen3-ASR audio encoder window from export_encoder.py,
# into an artifact. The phone's compiler can exhaust the phone's memory and
# panic its kernel (graphs below the size cap did), so this entry point checks
# everything itself, however it is called:
#  * KEY is one of the supported encoder windows (Qwen3-ASR 0.6B or 1.7B,
#    cpp/npu/encoder_contract.hpp), and TFLITE's single subgraph has exactly
#    that window's inputs and output; the artifact's I/O sizes follow from KEY;
#  * the model fingerprint, from the phone-side GGUF or given (0x..., from the
#    app's status-*.txt), equals EXPECT_HASH, the fingerprint of the checkpoint
#    TFLITE was exported from (export_encoder.py --fingerprint);
#  * the phone has at least NPU_MIN_MEM_MB (default 4096) available and TFLITE
#    is at most 640 MB.
# APPS: comma-separated packages; a debuggable one gets the artifact in
# files/npu (run-as), any other in its external files dir
# (/sdcard/Android/data/<pkg>/files/npu). Both are on the engine's
# STARLING_NPU_DIR (NativeSupport). Env: PY (python3 for the graph check).
set -euo pipefail
[ $# -eq 8 ] || { sed -n 2,19p "$0" >&2; exit 2; }
NPU_TOOL=$1; TFL=$2; SRC=$3; EXPECT=$4; KEY=$5; DIR=$6; APPS=$7; T=$8
PY=${PY:-python3}
IFS=, read -r -a PKGS <<<"$APPS"
for APP in "${PKGS[@]}"; do
  [[ $APP =~ ^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)+$ ]] || { echo "bad package name '$APP'" >&2; exit 2; }
done

# One argument quoted for the phone's shell.
q() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
# A fingerprint in canonical form (0x + 16 lower-case hex digits), or "".
canon() {
  local h
  h=$(printf '%s' "$1" | tr 'A-FX' 'a-fx')
  [[ $h =~ ^0x[0-9a-f]{1,16}$ ]] || { echo ""; return; }
  h=${h#0x}
  printf '0x%s%s' "$(printf '%*s' $((16 - ${#h})) '' | tr ' ' 0)" "$h"
}

# ---- the graph: a supported encoder window, and nothing else ----
case $KEY in
  "qwen3-asr-enc/1 mels=128 frames=800 tokens=104 d=896 L=18 H=14 ff=3584 ds=480 out=1024") OUTD=1024 ;;
  "qwen3-asr-enc/1 mels=128 frames=800 tokens=104 d=1024 L=24 H=16 ff=4096 ds=480 out=2048") OUTD=2048 ;;
  *) echo "refusing '$KEY': only the Qwen3-ASR 0.6B/1.7B encoder windows compile safely on the phone" >&2; exit 2 ;;
esac
MELS=128 FRAMES=800 TOKENS=104
IN=$((MELS * FRAMES * 4)),$((TOKENS * 4))
OUT=$((TOKENS * OUTD * 4))
SIZE=$(stat -c %s "$TFL")
[ "$SIZE" -le $((640 << 20)) ] || { echo "graph is $((SIZE >> 20)) MB, above the 640 MB safety limit" >&2; exit 1; }
# The .tflite's own inputs and output (flatbuffer: Model.subgraphs ->
# SubGraph.{tensors, inputs, outputs} -> Tensor.{shape, type}).
"$PY" -I - "$TFL" "$MELS" "$FRAMES" "$TOKENS" "$OUTD" <<'EOF' || exit 2
import mmap
import struct
import sys

path, mels, frames, tokens, outd = sys.argv[1], *map(int, sys.argv[2:])


def fail(msg):
    sys.exit(f"refusing {path}: {msg}")


with open(path, "rb") as f:
    if f.seek(0, 2) < 8:
        fail("not a .tflite")
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


def u(fmt, o):
    if o < 0 or o + struct.calcsize(fmt) > len(m):
        fail("malformed flatbuffer")
    return struct.unpack_from(fmt, m, o)[0]


def field(t, i):  # position of field i of the table at t, or None
    vt = t - u("<i", t)
    if 4 + 2 * i >= u("<H", vt):
        return None
    o = u("<H", vt + 4 + 2 * i)
    return t + o if o else None


def vec(p):  # (length, first element) of the vector a field points to
    v = p + u("<I", p)
    return u("<I", v), v + 4


def tables(p):
    n, e = vec(p)
    return [e + 4 * k + u("<I", e + 4 * k) for k in range(n)]


def ints(p):
    n, e = vec(p)
    return [u("<i", e + 4 * k) for k in range(n)]


if len(m) < 8 or m[4:8] != b"TFL3":
    fail("not a .tflite")
root = u("<I", 0)
sg = field(root, 2)
subgraphs = tables(sg) if sg else []
if len(subgraphs) != 1:
    fail(f"{len(subgraphs)} subgraphs, expected 1")
s = subgraphs[0]
tp = field(s, 0)
tensors = tables(tp) if tp else []


def io(i):
    p = field(s, i)
    out = []
    for t in ints(p) if p else []:
        if not 0 <= t < len(tensors):
            fail("tensor index out of range")
        sp, ty = field(tensors[t], 0), field(tensors[t], 1)
        out.append((ints(sp) if sp else [], u("<b", ty) if ty else 0))   # type 0: FLOAT32
    return out


want_in = [([1, mels, frames], 0), ([1, 1, 1, tokens], 0)]
want_out = [([1, tokens, outd], 0)]
got_in, got_out = io(1), io(2)
if got_in != want_in or got_out != want_out:
    fail(f"inputs {got_in} / outputs {got_out} are not the encoder window's {want_in} / {want_out}")
EOF

# ---- the model it belongs to ----
EXP=$(canon "$EXPECT")
[ -n "$EXP" ] || { echo "bad expected fingerprint '$EXPECT'" >&2; exit 2; }
adb shell "mkdir -p $(q "$T") $(q "$DIR")" </dev/null
adb push -q "$NPU_TOOL" "$T/starling-npu-tool" </dev/null >/dev/null
TOOL=$(q "$T/starling-npu-tool")
case $SRC in
  0x*|0X*) HASH=$SRC ;;
  *) HASH=$(adb shell "$TOOL fingerprint $(q "$SRC")" </dev/null | tr -d "\r" || true) ;;
esac
HC=$(canon "$HASH")
[ -n "$HC" ] || { echo "fingerprint failed: $HASH" >&2; exit 1; }
if [ "$HC" != "$EXP" ]; then
  echo "refusing: the app's model ($HC) is not the checkpoint the graph was exported from ($EXP)." >&2
  echo "  A fine-tune or other weights with the same shapes would get the wrong encoder." >&2
  exit 1
fi
NAME="qwen3-asr-enc-${HC#0x}.stnpu"
echo "fingerprint $HC -> $NAME"

# ---- compile ----
AVAIL=$(adb shell "grep MemAvailable /proc/meminfo" </dev/null | tr -dc '0-9')
[ "${AVAIL:-0}" -ge $((${NPU_MIN_MEM_MB:-4096} * 1024)) ] ||
  { echo "phone has $((${AVAIL:-0} / 1024)) MB available, below ${NPU_MIN_MEM_MB:-4096} MB: not compiling now" >&2; exit 1; }
echo "MemAvailable $((AVAIL / 1024)) MB"
adb push -q "$TFL" "$T/graph.tflite" </dev/null >/dev/null
ART=$(q "$DIR/$NAME")
adb shell "cd $(q "$T") && ./starling-npu-tool compile graph.tflite $ART --in $IN --out $OUT --key $(q "$KEY") --hash $HC" </dev/null
adb shell "rm -f $(q "$T/graph.tflite"); $TOOL info $ART" </dev/null

# ---- install ----
for APP in "${PKGS[@]}"; do
  if adb shell "run-as $APP true" </dev/null >/dev/null 2>&1; then
    adb shell "run-as $APP mkdir -p files/npu && cat $ART | run-as $APP sh -c 'cat > files/npu/$NAME.tmp && mv files/npu/$NAME.tmp files/npu/$NAME'" </dev/null
    echo "installed for $APP (files/npu, run-as)"
  else
    X=/sdcard/Android/data/$APP/files/npu
    adb shell "mkdir -p $X && cp $ART $X/$NAME.tmp && mv $X/$NAME.tmp $X/$NAME" </dev/null
    echo "installed for $APP ($X)"
  fi
done
