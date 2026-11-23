"""Translate source-built NGG vertex shaders and check their controls."""

from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from vertex_aot import lower

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
    program = PROLOGUE + PRIMITIVE + ["s_endpgm"]
    # Allocation and primitives pass; the program stops at the missing vertex exports.
    expect_error(program, "Expected position exports")
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
    expect_error(PROLOGUE[:1] + PROLOGUE[4:] + PRIMITIVE + ["s_endpgm"], "Expected one allocation from wave 0")
    # A barrier that only wave 0 passes would hang the subgroup.
    wave0_barrier = PROLOGUE[:8] + ["s_barrier"] + PROLOGUE[8:] + PRIMITIVE + ["s_endpgm"]
    expect_error(replace(wave0_barrier, "s_cbranch_scc1 7", "s_cbranch_scc1 8"),
                 "Inconsistent subgroup barriers")
    expect_error(PROLOGUE, "Expected a single trailing s_endpgm")


def primitive_controls():
    program = PROLOGUE + PRIMITIVE + ["s_endpgm"]
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
    expect_error(PRIMITIVE + PROLOGUE + ["s_endpgm"], "Export before allocation")
    expect_error(replace(program, "v_mbcnt_hi_u32_b32 v9, -1, v9", "v_mbcnt_hi_u32_b32 v9, -1, v8"),
                 "Read of undefined vector register v8")


def source():
    prologue_controls()
    primitive_controls()


if __name__ == "__main__":
    source()
