"""Translate a gfx1013 NGG vertex shader to a desktop HLSL vertex shader.

The input is a wave64 vertex shader that runs as the ES half of an NGG primitive
shader. The subgroup's input vertex and primitive counts arrive in s2, each wave's
counts and index in s3, the primitive's vertex indices in v0 and v1 and VertexID in
v5. The primitive-shader prologue is accepted only when it behaves like a
pass-through vertex shader in every modeled subgroup, so that the desktop input
assembler can replace it.
"""

import argparse
import re
from pathlib import Path

from pixel_aot import container_code, disassemble
from resource_inventory import parts

U32 = (1 << 32) - 1
WAVE = (1 << 64) - 1

# ESVertCount and GSPrimCount of each wave in a modeled subgroup: a single triangle,
# one vertex, partial and full waves, a wave without primitives and a third wave.
SUBGROUPS = (
    ((3, 1),),
    ((1, 1),),
    ((17, 5),),
    ((64, 64),),
    ((64, 30), (6, 0)),
    ((64, 64), (64, 64), (40, 20)),
)

# Source width, destination width, result and whether SCC is the result's non-zero test.
SCALAR_ALU = {
    "s_mov_b64": (64, 64, lambda a: a, False),
    "s_and_b32": (32, 32, lambda a, b: a & b, True),
    "s_or_b32": (32, 32, lambda a, b: a | b, True),
    "s_lshl_b32": (32, 32, lambda a, b: a << (b & 31) & U32, True),
    "s_bfe_u32": (32, 32, lambda a, b: a >> (b & 31) & (1 << (b >> 16 & 127)) - 1, True),
}
SCALAR_COMPARE = {
    "s_cmp_eq_u32": lambda a, b: a == b,
    "s_cmp_lg_u32": lambda a, b: a != b,
}
# Lane-wise integer operations with concrete operands.
VECTOR_ALU = {
    "v_mbcnt_lo_u32_b32": lambda lane, a, b: (a & (1 << min(lane, 32)) - 1).bit_count() + b & U32,
    "v_mbcnt_hi_u32_b32": lambda lane, a, b: (a & (1 << max(lane - 32, 0)) - 1).bit_count() + b & U32,
}
VECTOR_COMPARE = {
    "v_cmp_gt_u32_e32": lambda a, b: a > b,
}
EXPORT = re.compile(r"(prim|pos0|param\d+) (v\d+|off), (v\d+|off), (v\d+|off), (v\d+|off)((?: [a-z]+)*)")
FLOAT_INLINE = {
    "0.5": 0x3F000000, "-0.5": 0xBF000000, "1.0": 0x3F800000, "-1.0": 0xBF800000,
    "2.0": 0x40000000, "-2.0": 0xC0000000, "4.0": 0x40800000, "-4.0": 0xC0800000,
}


def constant(token, width=32):
    """Bit pattern of an inline or literal constant at the operand width."""
    if token in FLOAT_INLINE and width == 32:
        return FLOAT_INLINE[token]
    if re.fullmatch(r"-?(?:0x[0-9a-f]+|\d+)", token):
        value = int(token, 0)
        if width == 64 and not -16 <= value <= 64:
            raise ValueError(f"Unsupported 64-bit literal {token}")
        return value & ((1 << width) - 1)
    raise ValueError(f"Unsupported operand {token}")


def pair(token):
    m = re.fullmatch(r"s\[(\d+):(\d+)\]", token)
    if not m or int(m[1]) % 2 or int(m[2]) != int(m[1]) + 1:
        raise ValueError(f"Unsupported scalar pair {token}")
    return int(m[1])


def subgroup(waves):
    """Entry SGPRs and per-lane VGPRs of each wave in one modeled subgroup.

    s2 (gs_tg_info) holds the subgroup's vertex count in bits 20:12 and primitive
    count in bits 30:22. s3 (merged_wave_info) holds the wave's ESVertCount in bits
    7:0, GSPrimCount in bits 15:8, its index in bits 27:24 and the wave count in bits
    31:28. A primitive thread receives its three vertex indices packed ten bits apart
    in v0. Lanes without a primitive hold distinct values that no check accepts.
    """
    vertices = sum(es for es, _ in waves)
    primitives = sum(gs for _, gs in waves)
    s2 = vertices << 12 | primitives << 22
    first = 0
    result = []
    for index, (es, gs) in enumerate(waves):
        s3 = es | gs << 8 | index << 24 | len(waves) << 28
        lanes = []
        expected = {}
        for lane in range(64):
            if lane < gs:
                p = first + lane
                indices = [(3 * p + 7 * k + 1) % vertices for k in range(3)]
                v0 = indices[0] | indices[1] << 10 | indices[2] << 20
                expected[lane] = v0
            else:
                v0 = 0xDEA00000 | index << 8 | lane
            lanes.append({0: v0, 1: 0xB1000000 | index << 8 | lane})
        result.append(({2: s2, 3: s3}, lanes, expected))
        first += gs
    return vertices, primitives, result


class Wave:
    """One wave64 of a modeled subgroup with concrete control state."""

    def __init__(self, index, sgprs, lanes):
        self.index = index
        self.s = dict(sgprs)
        self.v = lanes
        self.exec = WAVE
        self.vcc = None
        self.scc = None
        self.m0 = None
        self.barriers = []
        self.allocations = []
        self.exports = {}

    def sgpr(self, i):
        if i not in self.s:
            raise ValueError(f"Read of undefined scalar register s{i}")
        return self.s[i]

    def scalar(self, token, width=32):
        if token.startswith("s["):
            if width != 64:
                raise ValueError(f"Unexpected scalar pair {token}")
            lo = pair(token)
            return self.sgpr(lo) | self.sgpr(lo + 1) << 32
        if m := re.fullmatch(r"s(\d+)", token):
            if width != 32:
                raise ValueError(f"Unexpected scalar register {token}")
            return self.sgpr(int(m[1]))
        if token == "exec":
            if width != 64:
                raise ValueError("Unexpected 32-bit exec read")
            return self.exec
        if token in ("vcc", "m0"):
            value = getattr(self, token)
            if value is None:
                raise ValueError(f"Read of undefined {token}")
            if (token == "vcc") != (width == 64):
                raise ValueError(f"Unexpected {token} width")
            return value
        return constant(token, width)

    def put(self, dest, value, width):
        if dest == "exec" and width == 64:
            self.exec = value & WAVE
        elif dest == "vcc" and width == 64:
            self.vcc = value & WAVE
        elif dest == "m0" and width == 32:
            self.m0 = value & U32
        elif dest.startswith("s[") and width == 64:
            lo = pair(dest)
            self.s[lo], self.s[lo + 1] = value & U32, value >> 32 & U32
        elif re.fullmatch(r"s\d+", dest) and width == 32:
            self.s[int(dest[1:])] = value & U32
        else:
            raise ValueError(f"Unsupported scalar destination {dest}")

    def lane_value(self, token, lane):
        if m := re.fullmatch(r"v(\d+)", token):
            if int(m[1]) not in self.v[lane]:
                raise ValueError(f"Read of undefined vector register {token}")
            return self.v[lane][int(m[1])]
        return self.scalar(token)

    def live(self):
        return [lane for lane in range(64) if self.exec >> lane & 1]

    def vector_op(self, op, args):
        if op in VECTOR_COMPARE:
            dest, a, b = args
            if dest != "vcc":
                raise ValueError(f"Unsupported comparison destination {dest}")
            mask = 0
            for lane in self.live():
                x, y = self.lane_value(a, lane), self.lane_value(b, lane)
                if type(x) is not int or type(y) is not int:
                    raise ValueError("Per-vertex value in a predicate")
                mask |= VECTOR_COMPARE[op](x, y) << lane
            self.vcc = mask
            return
        if op not in VECTOR_ALU:
            raise ValueError(f"Unsupported vertex instruction {op}")
        dest, *sources = args
        if not re.fullmatch(r"v\d+", dest):
            raise ValueError(f"Unsupported vector destination {dest}")
        for lane in self.live():
            values = [self.lane_value(t, lane) for t in sources]
            if any(type(value) is not int for value in values):
                raise ValueError(f"Per-vertex operand of {op}")
            self.v[lane][int(dest[1:])] = VECTOR_ALU[op](lane, *values)

    def export(self, tail):
        m = EXPORT.fullmatch(tail)
        if not m:
            raise ValueError(f"Unsupported export {tail}")
        target, operands, flags = m[1], m.groups()[1:5], m[6].split()
        if target != "prim":
            raise ValueError(f"Unsupported export {tail}")
        if flags != ["done"] or operands[0] == "off" or operands[1:] != ("off",) * 3:
            raise ValueError("Unsupported primitive export")
        if self.index == 0 and not self.allocations:
            raise ValueError("Export before allocation")
        lanes = self.exports.setdefault(target, {})
        for lane in self.live():
            if lane in lanes:
                raise ValueError(f"Repeated {target} export")
            lanes[lane] = tuple(None if t == "off" else self.lane_value(t, lane) for t in operands)

    def scalar_op(self, op, args):
        if op in SCALAR_COMPARE:
            a, b = (self.scalar(t) for t in args)
            self.scc = SCALAR_COMPARE[op](a, b)
            return
        if op == "s_and_saveexec_b64":
            dest, source = args
            old = self.exec
            self.exec = old & self.scalar(source, 64)
            self.put(dest, old, 64)
            self.scc = self.exec != 0
            return
        if op not in SCALAR_ALU:
            raise ValueError(f"Unsupported vertex instruction {op}")
        source_width, dest_width, function, sets_scc = SCALAR_ALU[op]
        dest, *sources = args
        value = function(*(self.scalar(t, source_width) for t in sources))
        self.put(dest, value, dest_width)
        if sets_scc:
            self.scc = value != 0

    def branch(self, op, pc, tail, index):
        words = int(tail)
        target = pc + 4 + 4 * words
        if words < 0 or target not in index:
            raise ValueError("Expected a forward branch to an instruction")
        if op == "s_cbranch_scc1":
            if self.scc is None:
                raise ValueError("Undefined scalar predicate")
            taken = self.scc
        else:
            raise ValueError(f"Unsupported vertex instruction {op}")
        return index[target] if taken else None

    def run(self, program):
        index = {pc: i for i, (pc, _) in enumerate(program)}
        i = 0
        while True:
            pc, line = program[i]
            i += 1
            op, _, tail = line.partition(" ")
            args = tail.split(", ") if tail else []
            if op == "s_endpgm":
                return
            if op == "s_barrier":
                self.barriers.append(pc)
            elif op == "s_sendmsg":
                if tail != "sendmsg(MSG_GS_ALLOC_REQ)":
                    raise ValueError(f"Unsupported message {tail}")
                if self.m0 is None:
                    raise ValueError("Read of undefined m0")
                self.allocations.append(self.m0)
            elif op.startswith("s_cbranch_") or op == "s_branch":
                target = self.branch(op, pc, tail, index)
                if target is not None:
                    i = target
            elif op == "exp":
                self.export(tail)
            elif op.startswith("s_"):
                self.scalar_op(op, args)
            elif op.startswith("v_"):
                self.vector_op(op, args)
            else:
                raise ValueError(f"Unsupported vertex instruction {op}")


def check_subgroup(program, waves):
    """Run one modeled subgroup and check its synchronization, allocation and primitives."""
    vertices, primitives, entries = subgroup(waves)
    results = []
    for index, (sgprs, lanes, expected) in enumerate(entries):
        wave = Wave(index, sgprs, lanes)
        wave.run(program)
        results.append(wave)
    if any(wave.barriers != results[0].barriers for wave in results):
        raise ValueError("Inconsistent subgroup barriers")
    allocations = [(wave.index, m0) for wave in results for m0 in wave.allocations]
    if [index for index, _ in allocations] != [0]:
        raise ValueError("Expected one allocation from wave 0")
    if allocations[0][1] != vertices | primitives << 12:
        raise ValueError("Allocation differs from the subgroup's input counts")
    for wave, (_, _, expected) in zip(results, entries):
        primitive = wave.exports.get("prim", {})
        if primitive.keys() != expected.keys():
            raise ValueError("Primitive export lanes differ from the input primitives")
        if any(primitive[lane][0] != value for lane, value in expected.items()):
            raise ValueError("Primitive export differs from the input primitive")
    return results


def lower(program):
    """Return HLSL for a decoded NGG vertex program given as (byte offset, text) pairs."""
    if not program or program[-1][1] != "s_endpgm" or [line for _, line in program].count("s_endpgm") != 1:
        raise ValueError("Expected a single trailing s_endpgm")
    for waves in SUBGROUPS:
        check_subgroup(program, waves)
    raise ValueError("Expected position exports")


def translate(blob, library):
    code = container_code(parts(blob)[b"DXIL"], stage=1)
    return lower(disassemble(code, library, offsets=True, features=b"+wavefrontsize64,-wavefrontsize32"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("container", type=Path)
    p.add_argument("--llvm", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.suffix.lower() != ".hlsl" or a.output.resolve() == a.container.resolve():
        p.error("Expected a separate HLSL output")
    source = translate(a.container.read_bytes(), a.llvm)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(source, encoding="ascii")
