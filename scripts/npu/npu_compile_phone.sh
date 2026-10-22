#!/usr/bin/env bash
# Phone half of npu_compile.sh (run under the phone lock):
#   npu_compile_phone.sh NPU_TOOL TFLITE HASH KEY DIR APPS WORKDIR
# compiles TFLITE, a Qwen3-ASR audio encoder window from export_encoder.py
# or one Parakeet-TDT encoder piece from export_parakeet.py, into the artifact
# qwen3-asr-enc-<HASH>.stnpu or parakeet-enc-<HASH>-<piece>.stnpu. The phone's compiler can
# exhaust the phone's memory and panic its kernel (graphs below the size cap
# did), so this entry point checks everything itself, however it is called:
#  * KEY is one of the supported encoder windows (Qwen3-ASR 0.6B or 1.7B) or
#    Parakeet pieces (cpp/npu/encoder_contract.hpp), and TFLITE's single subgraph has exactly
#    that window's inputs and output and the operator signature (operator and
#    tensor counts, operators per builtin) export_encoder.py produces for it;
#    the artifact's I/O sizes follow from KEY;
#  * HASH (0x...) is the model hash export_encoder.py printed for the GGUF
#    TFLITE's weights came from (the digest of its encoder tensors and
#    metadata, which the engine computes from the GGUF it loads); it names
#    the artifact, and TFLITE must carry it: the exporter stamps the
#    model hash and KEY into the graph's metadata (starling_npu_provenance),
#    so another model's graph cannot be compiled under this one's hash;
#  * the phone has at least NPU_MIN_MEM_MB (default 4096) available and TFLITE
#    is no larger than its window's graph has been (Qwen3-ASR 0.6B: 640 MB;
#    1.7B, which exports with f32 weights: 1300 MB; Parakeet segment: 340 MB;
#    all compiled cleanly).
# APPS: comma-separated packages; a debuggable one gets the artifact in
# files/npu (run-as), any other in its external files dir
# (/sdcard/Android/data/<pkg>/files/npu). Both are on the engine's
# STARLING_NPU_DIR (NativeSupport). Env: PY (python3 for the graph check).
set -euo pipefail
[ $# -eq 7 ] || { sed -n 2,27p "$0" >&2; exit 2; }
NPU_TOOL=$1; TFL=$2; HASH=$3; KEY=$4; DIR=$5; APPS=$6; T=$7
PY=${PY:-python3}
IFS=, read -r -a PKGS <<<"$APPS"
for APP in "${PKGS[@]}"; do
  [[ $APP =~ ^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)+$ ]] || { echo "bad package name '$APP'" >&2; exit 2; }
done

# One argument quoted for the phone's shell (every byte kept, trailing
# newlines included: no command substitution).
q() { printf "'%s'" "${1//\'/\'\\\'\'}"; }
# A fingerprint in canonical form (0x + 16 lower-case hex digits), or "".
canon() {
  local h
  h=$(printf '%s' "$1" | tr 'A-FX' 'a-fx')
  [[ $h =~ ^0x[0-9a-f]{1,16}$ ]] || { echo ""; return; }
  h=${h#0x}
  printf '0x%s%s' "$(printf '%*s' $((16 - ${#h})) '' | tr ' ' 0)" "$h"
}

# ---- the model it belongs to ----
HC=$(canon "$HASH")
[ -n "$HC" ] && [ "$HC" != 0x0000000000000000 ] || { echo "bad model hash '$HASH'" >&2; exit 2; }

# ---- the graph: a supported encoder window, and nothing else ----
# GRAPH: what export_encoder.py produces for the window — tensor count,
# operator count and operators per TFLite builtin code (0 ADD, 3 CONV_2D,
# 4 DEPTHWISE_CONV_2D, 6 DEQUANTIZE, 9 FULLY_CONNECTED, 18 MUL, 22 RESHAPE,
# 25 SOFTMAX, 34 PAD, 39 TRANSPOSE, 40 MEAN, 41 SUB, 76 RSQRT,
# 99 SQUARED_DIFFERENCE, 126 BATCH_MATMUL, 150 GELU), so a different graph
# (a constant, a no-op) cannot pass under an encoder's key. 1.7B converts
# with lightweight_conversion and keeps f32 weights (no DEQUANTIZE).
# Measured on exports of qwen3-asr-0.6b-q4e8.gguf and
# qwen3-asr-1.7b-q4e8-imx.gguf with litert-torch 0.9.4 and ai-edge-quantizer
# 0.9.0 (the weights' storage type in the GGUF does not change the graph,
# only its constants); a converter that emits a different graph needs these
# updated (and the graph compiled once with care: see docs/fast-engine.md,
# Pixel TPU encoder).
PK="mels=128 chunk=26 rows=416 d=1024 L=24 H=8 ff=4096 ck=9 sc=256 jh=640"   # Parakeet-TDT 0.6B v3
PK_SEG="1463 1203 0:180 4:6 6:146 9:150 14:60 18:186 22:241 25:6 39:48 40:60 41:36 76:30 99:30 126:24"
case $KEY in
  "qwen3-asr-enc/2 mels=128 frames=800 tokens=104 d=896 L=18 H=14 ff=3584 ds=480 out=1024")
    MAXMB=640 SHAPES_IN="1,128,800;1,1,1,104" SHAPES_OUT="1,104,1024"
    GRAPH="1294 1046 0:129 3:2 4:1 6:114 9:111 18:74 22:277 25:18 34:3 39:74 40:74 41:37 76:37 99:37 126:36 150:22" ;;
  "qwen3-asr-enc/2 mels=128 frames=800 tokens=104 d=1024 L=24 H=16 ff=4096 ds=480 out=2048")
    MAXMB=1300 SHAPES_IN="1,128,800;1,1,1,104" SHAPES_OUT="1,104,2048"
    GRAPH="1871 1452 0:171 3:2 4:1 9:244 18:219 22:367 25:24 34:3 39:100 40:98 41:49 76:49 99:49 126:48 150:28" ;;
  # Parakeet's encoder window in five pieces (export_parakeet.py): the
  # subsampling chunk (a few MB) and four 6-layer segments (~320 MB each).
  "parakeet-enc/1 piece=sub $PK")
    MAXMB=16 SHAPES_IN="1,1,215,128;1,1,107,1;1,1,53,1;1,26,1" SHAPES_OUT="1,26,1024"
    GRAPH="46 24 0:1 3:2 4:3 6:6 9:1 18:3 22:4 34:3 39:1" ;;
  "parakeet-enc/1 piece=l0-6 $PK"|"parakeet-enc/1 piece=l6-12 $PK"|"parakeet-enc/1 piece=l12-18 $PK")
    MAXMB=340 SHAPES_IN="4,104,1024;4,104,1;1,1,1,416" SHAPES_OUT="4,104,1024" GRAPH=$PK_SEG ;;
  "parakeet-enc/1 piece=l18-24 $PK")
    MAXMB=340 SHAPES_IN="4,104,1024;4,104,1;1,1,1,416" SHAPES_OUT="4,104,640"
    GRAPH="1464 1203 0:179 4:6 6:147 9:151 14:60 18:185 22:241 25:6 39:48 40:60 41:36 76:30 99:30 126:24" ;;
  *) echo "refusing '$KEY': only the Qwen3-ASR 0.6B/1.7B encoder windows and Parakeet-TDT 0.6B's encoder pieces compile safely on the phone" >&2; exit 2 ;;
esac
# Byte sizes of the f32 inputs / output, from the shapes.
bytes() { local IFS=';' out="" s d n; for s in $1; do n=4; IFS=','; for d in $s; do n=$((n * d)); done; IFS=';'; out+="${out:+,}$n"; done; echo "$out"; }
IN=$(bytes "$SHAPES_IN")
OUT=$(bytes "$SHAPES_OUT")
case $KEY in
  parakeet-enc/*) PIECE=${KEY#* piece=}; PIECE=${PIECE%% *}; NAME="parakeet-enc-${HC#0x}-$PIECE.stnpu" ;;
  *) NAME="qwen3-asr-enc-${HC#0x}.stnpu" ;;
esac
SIZE=$(stat -c %s "$TFL")
[ "$SIZE" -le $((MAXMB << 20)) ] || { echo "graph is $((SIZE >> 20)) MB, above this window's $MAXMB MB safety limit" >&2; exit 1; }
# The .tflite's own inputs, output and operators (flatbuffer: Model.
# {operator_codes, subgraphs} -> SubGraph.{tensors, inputs, outputs,
# operators} -> Tensor.{shape, type}, Operator.opcode_index,
# OperatorCode.{deprecated_builtin_code, custom_code, builtin_code}).
# shellcheck disable=SC2086
"$PY" -I - "$TFL" "$HC" "$KEY" "$SHAPES_IN" "$SHAPES_OUT" $GRAPH <<'EOF' || exit 2
import mmap
import struct
import sys

path, want_hash, want_key = sys.argv[1:4]
shapes_in, shapes_out = sys.argv[4:6]
want_tensors, want_ops = map(int, sys.argv[6:8])
want_hist = {int(c): int(n) for c, n in (a.split(":") for a in sys.argv[8:])}
NAMES = {0: "ADD", 3: "CONV_2D", 4: "DEPTHWISE_CONV_2D", 6: "DEQUANTIZE", 9: "FULLY_CONNECTED",
         14: "LOGISTIC", 18: "MUL", 22: "RESHAPE", 25: "SOFTMAX", 34: "PAD", 39: "TRANSPOSE", 40: "MEAN", 41: "SUB",
         76: "RSQRT", 99: "SQUARED_DIFFERENCE", 126: "BATCH_MATMUL", 150: "GELU"}


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


def shapes(spec):   # "1,128,800;1,1,1,104" -> [([1, 128, 800], 0), ...] (type 0: FLOAT32)
    return [([int(d) for d in s.split(",")], 0) for s in spec.split(";")]


want_in, want_out = shapes(shapes_in), shapes(shapes_out)
got_in, got_out = io(1), io(2)
if got_in != want_in or got_out != want_out:
    fail(f"inputs {got_in} / outputs {got_out} are not the encoder window's {want_in} / {want_out}")


def opcode(oc):  # builtin code, or "custom"
    if field(oc, 1):
        return "custom"
    dep, bc = field(oc, 0), field(oc, 3)
    return max(u("<b", dep) if dep else 0, u("<i", bc) if bc else 0)


oc = field(root, 1)
codes = [opcode(t) for t in (tables(oc) if oc else [])]
op = field(s, 3)
hist = {}
for o in tables(op) if op else []:
    i = field(o, 0)
    k = u("<I", i) if i else 0
    if k >= len(codes):
        fail("operator code index out of range")
    hist[codes[k]] = hist.get(codes[k], 0) + 1
n_ops = sum(hist.values())
if (len(tensors), n_ops, hist) != (want_tensors, want_ops, want_hist):
    def show(h):
        order = sorted(h, key=lambda c: (isinstance(c, str), str(c).zfill(4)))
        return ", ".join(f"{NAMES.get(c, c)} {h[c]}" for c in order)
    fail(f"not the encoder graph export_encoder.py / export_parakeet.py builds: {len(tensors)} tensors, {n_ops} operators "
         f"({show(hist)}); expected {want_tensors} tensors, {want_ops} operators ({show(want_hist)})")


# The provenance export_encoder.py stamps (Model.metadata -> Metadata.{name,
# buffer} -> Model.buffers[buffer].data): exactly one, naming this model hash
# and key.
def blob(p):
    v = p + u("<I", p)
    n = u("<I", v)
    if v + 4 + n > len(m):
        fail("malformed flatbuffer")
    return bytes(m[v + 4:v + 4 + n])


md, bf = field(root, 6), field(root, 4)
buffers = tables(bf) if bf else []
stamps = []
for t in tables(md) if md else []:
    nm = field(t, 0)
    if nm and blob(nm) == b"starling_npu_provenance":
        bi = field(t, 1)
        k = u("<I", bi) if bi else 0
        if k >= len(buffers):
            fail("buffer index out of range")
        d = field(buffers[k], 0)
        stamps.append(blob(d) if d else b"")
if len(stamps) != 1:
    fail(f"{len(stamps)} provenance stamps, expected 1 (export the graph with this export_encoder.py)")
want = f"model_hash={want_hash}\nkey={want_key}\n".encode()
if stamps[0] != want:
    fail(f"the graph was exported for {stamps[0]!r}, not {want!r}: another model's graph cannot be "
         f"compiled under this model hash")
EOF

echo "model hash $HC -> $NAME"

# ---- compile ----
adb shell "mkdir -p $(q "$T") $(q "$DIR")" </dev/null
adb push -q "$NPU_TOOL" "$T/starling-npu-tool" </dev/null >/dev/null
# The tool links ggml-base (GGUF parsing); it sits next to the tool or, in a
# CMake build tree, under ggml/src.
BASE_LIB=
for c in "$(dirname "$NPU_TOOL")/libggml-base.so" "$(dirname "$NPU_TOOL")/ggml/src/libggml-base.so"; do
  [ -f "$c" ] && { BASE_LIB=$c; break; }
done
[ -n "$BASE_LIB" ] || { echo "no libggml-base.so next to $NPU_TOOL (or in its ggml/src)" >&2; exit 1; }
adb push -q "$BASE_LIB" "$T/libggml-base.so" </dev/null >/dev/null
TOOL="LD_LIBRARY_PATH=$(q "$T") $(q "$T/starling-npu-tool")"
AVAIL=$(adb shell "grep MemAvailable /proc/meminfo" </dev/null | tr -dc '0-9')
[ "${AVAIL:-0}" -ge $((${NPU_MIN_MEM_MB:-4096} * 1024)) ] ||
  { echo "phone has $((${AVAIL:-0} / 1024)) MB available, below ${NPU_MIN_MEM_MB:-4096} MB: not compiling now" >&2; exit 1; }
echo "MemAvailable $((AVAIL / 1024)) MB"
adb push -q "$TFL" "$T/graph.tflite" </dev/null >/dev/null
ART=$(q "$DIR/$NAME")
adb shell "cd $(q "$T") && LD_LIBRARY_PATH=. ./starling-npu-tool compile graph.tflite $ART --in $IN --out $OUT --key $(q "$KEY") --hash $HC" </dev/null
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
