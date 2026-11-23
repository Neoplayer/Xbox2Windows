"""Translate source-built NGG vertex shaders and check their controls."""

from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from vertex_aot import LAYOUTS, SUBGROUPS, Expressions, PrimitiveMismatch, check_subgroup, lower

# Wave 0 allocates the subgroup's input counts after a barrier that every wave passes.
PROLOGUE = [
    "s_barrier",
    "s_bfe_u32 s0, s3, 0x40018",
    "s_cmp_lg_u32 s0, 0",
    "s_cbranch_scc1 7",
    "s_bfe_u32 s1, s2, 0x9000c",
    "s_bfe_u32 s4, s2, 0x90016",
    "s_lshl_b32 s4, s4, 12",
    "s_or_b32 m0, s1, s4",
    "s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)",
]
# Primitive threads (lane < GSPrimCount) export the pass-through primitive in v0.
PRIMITIVE = [
    "v_mbcnt_lo_u32_b32 v9, -1, 0",
    "v_mbcnt_hi_u32_b32 v9, -1, v9",
    "s_bfe_u32 s5, s3, 0x80008",
    "v_cmp_gt_u32_e32 vcc, s5, v9",
    "s_and_saveexec_b64 s[6:7], vcc",
    "exp prim v0, off, off, off done",
    "s_mov_b64 exec, s[6:7]",
]
# Vertex threads (lane < ESVertCount) export a full-screen triangle position from VertexID:
# uv = ((id << 1) & 2, id & 2) and position = (2u - 1, 1 - 2v, 0, 1).
VERTEX = [
    "s_and_b32 s5, s3, 0xff",
    "v_cmp_gt_u32_e32 vcc, s5, v9",
    "s_and_saveexec_b64 s[6:7], vcc",
    "v_lshlrev_b32_e32 v10, 1, v5",
    "v_and_b32_e32 v10, 2, v10",
    "v_and_b32_e32 v11, 2, v5",
    "v_cvt_f32_u32_e32 v10, v10",
    "v_cvt_f32_u32_e32 v11, v11",
    "v_fma_f32 v12, v10, 2.0, -1.0",
    "v_fma_f32 v13, v11, -2.0, 1.0",
    "v_mov_b32_e32 v14, 0",
    "v_mov_b32_e32 v15, 1.0",
    "exp pos0 v12, v13, v14, v15 done",
    "exp param0 v10, v11, v14, v15",
]
FULLSCREEN_HLSL = """struct VertexOutput {
  float4 position : SV_Position;
  float4 attr0 : TEXCOORD0;
};

VertexOutput main(uint vertex_id : SV_VertexID) {
  uint t0 = (vertex_id << 1u);
  uint t1 = (0x2u & t0);
  uint t2 = (0x2u & vertex_id);
  precise float t3 = float(t1);
  precise float t4 = float(t2);
  precise float t5 = mad(t3, asfloat(0x40000000u), asfloat(0xbf800000u));
  precise float t6 = mad(t4, asfloat(0xc0000000u), asfloat(0x3f800000u));
  VertexOutput result;
  result.position = float4(t5, t6, asfloat(0x00000000u), asfloat(0x3f800000u));
  result.attr0 = float4(t3, t4, asfloat(0x00000000u), asfloat(0x3f800000u));
  return result;
}
"""
# The same triangle from a program that packs 16-bit input indices itself and sets EXEC
# the way LLVM does for merged shaders: s_bfm_b64 cannot express 64 lanes, so a full
# wave takes the s_cmov_b64 path. A wave without vertices skips to the end.
PACKED_PRIMITIVE = [
    "s_bfe_u32 s5, s3, 0x80008",
    "s_bfm_b64 exec, s5, 0",
    "s_cmp_eq_u32 s5, 64",
    "s_cmov_b64 exec, -1",
    "v_bfe_u32 v20, v0, 16, 16",
    "v_and_b32_e32 v21, 0xffff, v0",
    "v_and_b32_e32 v22, 0xffff, v1",
    "v_lshl_or_b32 v23, v20, 10, v21",
    "v_lshl_or_b32 v23, v22, 20, v23",
    "exp prim v23, off, off, off done",
    "s_mov_b64 exec, -1",
]
PACKED_VERTEX = [
    "v_mbcnt_lo_u32_b32 v9, -1, 0",
    "v_mbcnt_hi_u32_b32 v9, -1, v9",
    "s_and_b32 s5, s3, 0xff",
    "v_cmpx_gt_u32_e32 s5, v9",
    "s_cbranch_execz 18",
    "v_and_b32_e32 v10, 1, v5",
    "v_lshrrev_b32_e32 v11, 1, v5",
    "v_cvt_f32_u32_e32 v10, v10",
    "v_cvt_f32_i32_e32 v11, v11",
    "v_mov_b32_e32 v12, -1.0",
    "v_fmac_f32_e32 v12, 4.0, v10",
    "v_mul_f32_e32 v13, 4.0, v11",
    "v_sub_f32_e32 v13, 1.0, v13",
    "v_mov_b32_e32 v14, 0",
    "v_mov_b32_e32 v15, 1.0",
    "exp pos0 v12, v13, v14, v15 done",
    "v_add_f32_e32 v10, v10, v10",
    "v_add_f32_e32 v11, v11, v11",
    "exp param0 v10, v11, v14, v15",
    "exp param1 v11, off, v10, off",
]
PACKED_HLSL = """struct VertexOutput {
  float4 position : SV_Position;
  float4 attr0 : TEXCOORD0;
  float4 attr1 : TEXCOORD1;
};

VertexOutput main(uint vertex_id : SV_VertexID) {
  uint t0 = (0x1u & vertex_id);
  uint t1 = (vertex_id >> 1u);
  precise float t2 = float(t0);
  precise float t3 = float(asint(t1));
  precise float t4 = mad(asfloat(0x40800000u), t2, asfloat(0xbf800000u));
  precise float t5 = (asfloat(0x40800000u) * t3);
  precise float t6 = (asfloat(0x3f800000u) - t5);
  precise float t7 = (t2 + t2);
  precise float t8 = (t3 + t3);
  VertexOutput result;
  result.position = float4(t4, t6, asfloat(0x00000000u), asfloat(0x3f800000u));
  result.attr0 = float4(t7, t8, asfloat(0x00000000u), asfloat(0x3f800000u));
  result.attr1 = float4(t8, 0.0f, t7, 0.0f);
  return result;
}
"""
PASS_THROUGH = PROLOGUE + PRIMITIVE + VERTEX + ["s_endpgm"]
# The barrier is a Navi1x workaround; the second program omits it.
PACKED = PROLOGUE[1:] + PACKED_PRIMITIVE + PACKED_VERTEX + ["s_endpgm"]
FIXTURES = ((PASS_THROUGH, FULLSCREEN_HLSL), (PACKED, PACKED_HLSL))
VOP3 = {"v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32", "v_fma_f32", "v_bfe_u32", "v_lshl_or_b32"}
SOPP = {"s_barrier", "s_branch", "s_cbranch_scc1", "s_cbranch_execz", "s_endpgm", "s_sendmsg"}


def size(line):
    """Encoded length in bytes. The binary check compares these offsets with LLVM's."""
    op, _, tail = line.partition(" ")
    if op in SOPP:
        return 4
    base = 8 if op in VOP3 or op.endswith("_e64") or op == "exp" else 4
    literal = any(re.fullmatch(r"-?(?:0x[0-9a-f]+|\d+)", t) and not -16 <= int(t, 0) <= 64
                  for t in tail.split(", "))
    return base + 4 * literal


def decoded(program):
    offsets, pc = [], 0
    for line in program:
        offsets.append((pc, line))
        pc += size(line)
    return offsets


def replace(program, old, new):
    if program.count(old) != 1:
        raise AssertionError(f"Expected one {old!r}")
    return [new if line == old else line for line in program]


def expect_error(program, message):
    try:
        lower(decoded(program))
    except ValueError as error:
        if message not in str(error):
            raise AssertionError(f"Expected {message!r}, got {error!r}") from None
        return
    raise AssertionError(f"Accepted a program that should fail with {message!r}")


def prologue_controls():
    expect_error(PROLOGUE + PRIMITIVE + ["s_endpgm"], "Position export lanes differ from the input vertices")
    program = PROLOGUE + PRIMITIVE + VERTEX + ["s_endpgm"]
    expect_error(replace(program, "s_cmp_lg_u32 s0, 0", "s_cmp_lg_u32 s0, 1"),
                 "Export before allocation")
    expect_error(replace(program, "s_cbranch_scc1 7", "s_cbranch_scc1 0"),
                 "Expected one allocation from wave 0")
    expect_error(replace(program, "s_bfe_u32 s4, s2, 0x90016", "s_bfe_u32 s4, s3, 0x80008"),
                 "Allocation differs from the subgroup's input counts")
    expect_error(replace(program, "s_lshl_b32 s4, s4, 12", "s_lshl_b32 s4, s4, 13"),
                 "Allocation differs from the subgroup's input counts")
    expect_error(replace(program, "s_bfe_u32 s1, s2, 0x9000c", "s_bfe_u32 s1, s8, 0x9000c"),
                 "Read of undefined scalar register s8")
    expect_error(replace(program, "s_cbranch_scc1 7", "s_cbranch_scc1 -3"),
                 "Expected a forward branch")
    expect_error(replace(program, "s_barrier", "s_nop 0"), "Unsupported vertex instruction s_nop")
    expect_error(PROLOGUE[:1] + PROLOGUE[4:] + PRIMITIVE + VERTEX + ["s_endpgm"],
                 "Expected one allocation from wave 0")
    # A barrier that only wave 0 passes would hang the subgroup.
    wave0_barrier = PROLOGUE[:8] + ["s_barrier"] + PROLOGUE[8:] + PRIMITIVE + VERTEX + ["s_endpgm"]
    expect_error(replace(wave0_barrier, "s_cbranch_scc1 7", "s_cbranch_scc1 8"),
                 "Inconsistent subgroup barriers")
    expect_error(PROLOGUE, "Expected a single trailing s_endpgm")


def primitive_controls():
    program = PROLOGUE + PRIMITIVE + VERTEX + ["s_endpgm"]
    export = "exp prim v0, off, off, off done"
    expect_error(replace(program, "s_bfe_u32 s5, s3, 0x80008", "s_bfe_u32 s5, s3, 0x80000"),
                 "Primitive export lanes differ from the input primitives")
    expect_error([line for line in program if line != "s_and_saveexec_b64 s[6:7], vcc"
                  and line != "s_mov_b64 exec, s[6:7]"],
                 "Primitive export lanes differ from the input primitives")
    for operand in ("v9", "v1"):
        expect_error(replace(program, export, f"exp prim {operand}, off, off, off done"),
                     "Primitive export differs from the input primitive")
    expect_error(replace(program, export, "exp prim v2, off, off, off done"),
                 "Read of undefined vector register v2")
    expect_error(replace(program, export, "exp prim v0, off, off, off"), "Unsupported primitive export")
    expect_error(replace(program, export, "exp prim v0, v0, off, off done"), "Unsupported primitive export")
    expect_error(program[:-2] + [export] + program[-2:], "Repeated prim export")
    expect_error(PRIMITIVE + PROLOGUE + VERTEX + ["s_endpgm"], "Export before allocation")
    expect_error(replace(program, "v_mbcnt_hi_u32_b32 v9, -1, v9", "v_mbcnt_hi_u32_b32 v9, -1, v8"),
                 "Read of undefined vector register v8")


def position_controls():
    program = PROLOGUE + PRIMITIVE + VERTEX + ["s_endpgm"]
    export = "exp pos0 v12, v13, v14, v15 done"
    for old, new in (("v_lshlrev_b32_e32 v10, 1, v5", "v_lshlrev_b32_e32 v10, 1, v9"),
                     ("v_and_b32_e32 v11, 2, v5", "v_and_b32_e32 v11, 2, v0"),
                     ("v_mov_b32_e32 v14, 0", "v_mov_b32_e32 v14, s3")):
        expect_error(replace(program, old, new), "Position depends on more than VertexID")
    expect_error(replace(program, "v_and_b32_e32 v11, 2, v5", "v_and_b32_e32 v11, 2, v8"),
                 "Read of undefined vector register v8")
    expect_error(replace(program, "s_and_b32 s5, s3, 0xff", "s_bfe_u32 s5, s3, 0x80008"),
                 "Position export lanes differ from the input vertices")
    expect_error(program[:len(PROLOGUE + PRIMITIVE)] + program[len(PROLOGUE + PRIMITIVE) + 3:],
                 "Position export lanes differ from the input vertices")
    expect_error(replace(program, export, "exp pos0 v12, v13, v14, v15"), "Unsupported position export")
    expect_error(replace(program, export, "exp pos0 v12, v13, off, v15 done"), "Unsupported position export")
    expect_error(replace(program, export, "exp pos1 v12, v13, v14, v15 done"), "Unsupported export")
    expect_error(program[:-2] + [export] + program[-2:], "Repeated pos0 export")
    expect_error(replace(program, "v_mov_b32_e32 v14, 0", "v_cmp_gt_u32_e32 vcc, 2, v5"),
                 "Per-vertex value in a predicate")
    expect_error(replace(program, "v_mov_b32_e32 v14, 0", "v_mbcnt_lo_u32_b32 v14, v5, 0"),
                 "Per-vertex operand of v_mbcnt_lo_u32_b32")
    expect_error(replace(program, "v_mov_b32_e32 v14, 0", "v_xor_b32_e32 v14, 1, v10"),
                 "Unsupported vertex instruction v_xor_b32_e32")


def parameter_controls():
    program = PROLOGUE + PRIMITIVE + VERTEX + ["s_endpgm"]
    export = "exp param0 v10, v11, v14, v15"
    partial = lower(decoded(replace(program, export, "exp param3 v10, off, v11, off")))
    if "  result.attr3 = float4(t3, 0.0f, t4, 0.0f);\n" not in partial or "TEXCOORD3;" not in partial:
        raise AssertionError(f"Unexpected partial parameter:\n{partial}")
    expect_error(replace(program, export, "exp param0 v9, v11, v14, v15"), "param0 depends on more than VertexID")
    narrowed = ["v_cmp_gt_u32_e32 vcc, 2, v9", "s_and_saveexec_b64 s[8:9], vcc"]
    expect_error(program[:-2] + narrowed + program[-2:], "param0 export lanes differ from the input vertices")
    expect_error(program[:-2] + ["s_mov_b64 exec, -1"] + program[-2:], "Read of undefined vector register v10")
    expect_error(replace(program, export, "exp param0 v10, v11, v14, v15 done"), "Unsupported parameter export")
    expect_error(replace(program, export, "exp param0 off, off, off, off"), "Unsupported parameter export")
    expect_error(replace(program, export, "exp param32 v10, v11, v14, v15"), "Unsupported export")
    expect_error(program[:-1] + [export, "s_endpgm"], "Repeated param0 export")


def packed_controls():
    program = PACKED
    for old, new in (("v_lshl_or_b32 v23, v20, 10, v21", "v_lshl_or_b32 v23, v20, 11, v21"),
                     ("v_and_b32_e32 v22, 0xffff, v1", "v_bfe_u32 v22, v1, 16, 16"),
                     ("exp prim v23, off, off, off done", "exp prim v21, off, off, off done")):
        expect_error(replace(program, old, new), "Primitive export differs from the input primitive")
    # Without the s_cmov_b64 path a full wave exports no primitive; without the comparison
    # s_cmov_b64 reads the s_bfe_u32 SCC and every lane exports one.
    expect_error([line for line in program if line not in ("s_cmp_eq_u32 s5, 64", "s_cmov_b64 exec, -1")],
                 "Primitive export lanes differ from the input primitives")
    expect_error([line for line in program if line != "s_cmp_eq_u32 s5, 64"],
                 "Primitive export lanes differ from the input primitives")
    expect_error(["s_cmov_b64 exec, -1"] + program, "Undefined scalar predicate")
    expect_error(replace(program, "v_cmpx_gt_u32_e32 s5, v9", "v_cmpx_gt_u32_e32 s5, v5"),
                 "Per-vertex value in a predicate")
    expect_error(replace(program, "v_lshrrev_b32_e32 v11, 1, v5", "v_lshrrev_b32_e32 v11, 1, v1"),
                 "Position depends on more than VertexID")


def layout_controls():
    """Each fixture reproduces the input primitive in its own layout only."""
    for program, layout in ((PASS_THROUGH, LAYOUTS[0]), (PACKED, LAYOUTS[1])):
        for other in LAYOUTS:
            try:
                for shape in SUBGROUPS:
                    check_subgroup(decoded(program), shape, other, Expressions())
            except PrimitiveMismatch:
                if other == layout:
                    raise
            else:
                if other != layout:
                    raise AssertionError(f"Primitive export also matched the {other} layout")


def source():
    prologue_controls()
    primitive_controls()
    position_controls()
    parameter_controls()
    packed_controls()
    layout_controls()
    for program, expected in FIXTURES:
        translated = lower(decoded(program))
        if translated != expected:
            raise AssertionError(f"Unexpected HLSL:\n{translated}")
    return FIXTURES


if __name__ == "__main__":
    source()
