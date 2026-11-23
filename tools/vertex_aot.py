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
import struct
from dataclasses import dataclass
from pathlib import Path

from pixel_aot import container_code, disassemble
from resource_inventory import parts

U32 = (1 << 32) - 1
WAVE = (1 << 64) - 1

# ESVertCount and GSPrimCount of each wave in a modeled subgroup: a single triangle,
# one vertex, partial and full waves, waves without primitives or without vertices
# and a third wave.
SUBGROUPS = (
    ((3, 1),),
    ((1, 1),),
    ((17, 5),),
    ((64, 64),),
    ((64, 30), (6, 0)),
    ((64, 64), (0, 36)),
    ((64, 64), (64, 64), (40, 20)),
)
# Primitive input layouts. In pass-through mode v0 holds the packed export value; otherwise
# v0 and v1 hold 16-bit vertex indices that the shader packs itself.
LAYOUTS = ("pass-through", "16-bit indices")

# Source width, destination width, result and whether SCC is the result's non-zero test.
SCALAR_ALU = {
    "s_mov_b64": (64, 64, lambda a: a, False),
    "s_and_b32": (32, 32, lambda a, b: a & b, True),
    "s_or_b32": (32, 32, lambda a, b: a | b, True),
    "s_lshl_b32": (32, 32, lambda a, b: a << (b & 31) & U32, True),
    "s_bfe_u32": (32, 32, lambda a, b: a >> (b & 31) & (1 << (b >> 16 & 127)) - 1, True),
    "s_bfm_b64": (32, 64, lambda a, b: ((1 << (a & 63)) - 1) << (b & 63) & WAVE, False),
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
    "v_cmpx_gt_u32_e32": lambda a, b: a > b,
}
EXPORT = re.compile(r"(prim|pos0|param\d+) (v\d+|off), (v\d+|off), (v\d+|off), (v\d+|off)((?: [a-z]+)*)")
FLOAT_INLINE = {
    "0.5": 0x3F000000, "-0.5": 0xBF000000, "1.0": 0x3F800000, "-1.0": 0xBF800000,
    "2.0": 0x40000000, "-2.0": 0xC0000000, "4.0": 0x40800000, "-4.0": 0xC0800000,
}


@dataclass(frozen=True)
class Sym:
    """A per-vertex value held in an HLSL variable of type uint or float."""

    name: str
    kind: str


VERTEX_ID = Sym("vertex_id", "uint")


def as_uint(value):
    if isinstance(value, Sym):
        return value.name if value.kind == "uint" else f"asuint({value.name})"
    return f"{value:#x}u"


def as_float(value):
    if isinstance(value, Sym):
        return value.name if value.kind == "float" else f"asfloat({value.name})"
    return f"asfloat({value:#010x}u)"


def amount(value):
    """Shift operand: the hardware reads bits 4:0."""
    return f"{value & 31}u" if type(value) is int else f"({as_uint(value)} & 31u)"


def f32_bits(number):
    return struct.unpack("<I", struct.pack("<f", number))[0]


# Integer and conversion results fold for concrete operands and otherwise become
# expressions. Floating-point results are always expressions, evaluated by DXC.
INTEGER = {
    "v_lshlrev_b32_e32": (lambda a, b: b << (a & 31) & U32, lambda a, b: f"({as_uint(b)} << {amount(a)})"),
    "v_lshrrev_b32_e32": (lambda a, b: b >> (a & 31), lambda a, b: f"({as_uint(b)} >> {amount(a)})"),
    "v_and_b32_e32": (lambda a, b: a & b, lambda a, b: f"({as_uint(a)} & {as_uint(b)})"),
    "v_bfe_u32": (lambda a, b, c: a >> (b & 31) & (1 << (c & 31)) - 1,
                  lambda a, b, c: f"(({as_uint(a)} >> {amount(b)}) & ((1u << {amount(c)}) - 1u))"),
    "v_lshl_or_b32": (lambda a, b, c: (a << (b & 31) | c) & U32,
                      lambda a, b, c: f"(({as_uint(a)} << {amount(b)}) | {as_uint(c)})"),
}
CONVERT = {
    "v_cvt_f32_u32_e32": (lambda a: float(a), lambda a: f"float({as_uint(a)})"),
    "v_cvt_f32_i32_e32": (lambda a: float(a - (a >> 31 << 32)), lambda a: f"float(asint({as_uint(a)}))"),
}
# v_fmac_f32 reads its destination as the third operand.
FLOAT = {
    "v_fma_f32": lambda a, b, c: f"mad({as_float(a)}, {as_float(b)}, {as_float(c)})",
    "v_fmac_f32_e32": lambda a, b, c: f"mad({as_float(a)}, {as_float(b)}, {as_float(c)})",
    "v_mul_f32_e32": lambda a, b: f"({as_float(a)} * {as_float(b)})",
    "v_add_f32_e32": lambda a, b: f"({as_float(a)} + {as_float(b)})",
    "v_sub_f32_e32": lambda a, b: f"({as_float(a)} - {as_float(b)})",
}


class PrimitiveMismatch(ValueError):
    """The primitive export differs from the input primitive in the modeled layout."""


class Expressions:
    """Per-vertex values in creation order. Equal expressions share one variable."""

    def __init__(self):
        self.names = {}
        self.entries = []

    def add(self, kind, text, operands):
        if (kind, text) not in self.names:
            sym = Sym(f"t{len(self.entries)}", kind)
            uses = {v for v in operands if isinstance(v, Sym) and v != VERTEX_ID}
            self.names[kind, text] = sym
            self.entries.append((sym, text, uses))
        return self.names[kind, text]

    def declarations(self, outputs):
        needed = {v for v in outputs if isinstance(v, Sym)}
        for sym, _, uses in reversed(self.entries):
            if sym in needed:
                needed |= uses
        return [f"  {'precise float' if sym.kind == 'float' else 'uint'} {sym.name} = {text};"
                for sym, text, _ in self.entries if sym in needed]


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


def subgroup(waves, layout):
    """Entry SGPRs and per-lane VGPRs of each wave in one modeled subgroup.

    s2 (gs_tg_info) holds the subgroup's vertex count in bits 20:12 and primitive
    count in bits 30:22. s3 (merged_wave_info) holds the wave's ESVertCount in bits
    7:0, GSPrimCount in bits 15:8, its index in bits 27:24 and the wave count in bits
    31:28. A primitive thread receives its three vertex indices packed ten bits apart
    in v0 in pass-through mode, or as 16-bit halves of v0 and v1 otherwise. Either way
    it must export the packed value. Lanes without a primitive hold distinct values
    that no check accepts.
    VertexID (v5) is the symbolic input of the translated shader in every lane.
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
                expected[lane] = indices[0] | indices[1] << 10 | indices[2] << 20
                if layout == "pass-through":
                    v0, v1 = expected[lane], 0xB1000000 | index << 8 | lane
                else:
                    v0, v1 = indices[0] | indices[1] << 16, indices[2] | (indices[0] + 1) % vertices << 16
            else:
                v0, v1 = 0xDEA00000 | index << 8 | lane, 0xB1000000 | index << 8 | lane
            lanes.append({0: v0, 1: v1, 5: VERTEX_ID})
        result.append(({2: s2, 3: s3}, lanes, expected, es))
        first += gs
    return vertices, primitives, result


class Wave:
    """One wave64 of a modeled subgroup with concrete control state."""

    def __init__(self, index, sgprs, lanes, expressions):
        self.index = index
        self.e = expressions
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
            # A gfx10 v_cmpx writes EXEC only.
            dest, a, b = ("exec", *args) if op.startswith("v_cmpx_") else args
            if dest not in ("vcc", "exec"):
                raise ValueError(f"Unsupported comparison destination {dest}")
            mask = 0
            for lane in self.live():
                x, y = self.lane_value(a, lane), self.lane_value(b, lane)
                if type(x) is not int or type(y) is not int:
                    raise ValueError("Per-vertex value in a predicate")
                mask |= VECTOR_COMPARE[op](x, y) << lane
            self.put(dest, mask, 64)
            return
        if op != "v_mov_b32_e32" and not any(op in table for table in (VECTOR_ALU, INTEGER, CONVERT, FLOAT)):
            raise ValueError(f"Unsupported vertex instruction {op}")
        dest, *sources = args
        if not re.fullmatch(r"v\d+", dest):
            raise ValueError(f"Unsupported vector destination {dest}")
        if op == "v_fmac_f32_e32":
            sources.append(dest)
        for lane in self.live():
            values = [self.lane_value(t, lane) for t in sources]
            self.v[lane][int(dest[1:])] = self.lane_result(op, lane, values)

    def lane_result(self, op, lane, values):
        if op == "v_mov_b32_e32":
            return values[0]
        concrete = all(type(value) is int for value in values)
        if op in VECTOR_ALU:
            if not concrete:
                raise ValueError(f"Per-vertex operand of {op}")
            return VECTOR_ALU[op](lane, *values)
        if op in INTEGER:
            fold, text = INTEGER[op]
            return fold(*values) if concrete else self.e.add("uint", text(*values), values)
        if op in CONVERT:
            fold, text = CONVERT[op]
            return f32_bits(fold(*values)) if concrete else self.e.add("float", text(*values), values)
        return self.e.add("float", FLOAT[op](*values), values)

    def export(self, tail):
        m = EXPORT.fullmatch(tail)
        if not m:
            raise ValueError(f"Unsupported export {tail}")
        target, operands, flags = m[1], m.groups()[1:5], m[6].split()
        if target == "prim":
            if flags != ["done"] or operands[0] == "off" or operands[1:] != ("off",) * 3:
                raise ValueError("Unsupported primitive export")
        elif target == "pos0":
            if flags != ["done"] or "off" in operands:
                raise ValueError("Unsupported position export")
        elif target.startswith("param") and int(target[5:]) < 32:
            if flags or operands == ("off",) * 4:
                raise ValueError("Unsupported parameter export")
        else:
            raise ValueError(f"Unsupported export {tail}")
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
        if op == "s_cmov_b64":
            if self.scc is None:
                raise ValueError("Undefined scalar predicate")
            if self.scc:
                self.put(args[0], self.scalar(args[1], 64), 64)
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
        elif op == "s_cbranch_execz":
            taken = self.exec == 0
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


def check_subgroup(program, waves, layout, expressions):
    """Run one modeled subgroup and check its synchronization, allocation and export lanes."""
    vertices, primitives, entries = subgroup(waves, layout)
    results = []
    for index, (sgprs, lanes, _, _) in enumerate(entries):
        wave = Wave(index, sgprs, lanes, expressions)
        wave.run(program)
        results.append(wave)
    if any(wave.barriers != results[0].barriers for wave in results):
        raise ValueError("Inconsistent subgroup barriers")
    allocations = [(wave.index, m0) for wave in results for m0 in wave.allocations]
    if [index for index, _ in allocations] != [0]:
        raise ValueError("Expected one allocation from wave 0")
    if allocations[0][1] != vertices | primitives << 12:
        raise ValueError("Allocation differs from the subgroup's input counts")
    for wave, (_, _, expected, wave_vertices) in zip(results, entries):
        primitive = wave.exports.get("prim", {})
        if primitive.keys() != expected.keys():
            raise ValueError("Primitive export lanes differ from the input primitives")
        if any(primitive[lane][0] != value for lane, value in expected.items()):
            raise PrimitiveMismatch("Primitive export differs from the input primitive")
        if wave.exports.get("pos0", {}).keys() != set(range(wave_vertices)):
            raise ValueError("Position export lanes differ from the input vertices")
    return results


def float4(values):
    return f"float4({', '.join('0.0f' if v is None else as_float(v) for v in values)})"


def emit(expressions, position, parameters):
    """PARAM k becomes TEXCOORD k, the input slot a translated pixel shader reads for attribute k."""
    lines = [
        "struct VertexOutput {",
        "  float4 position : SV_Position;",
        *(f"  float4 attr{k} : TEXCOORD{k};" for k in parameters),
        "};",
        "",
        "VertexOutput main(uint vertex_id : SV_VertexID) {",
        *expressions.declarations([*position, *(v for values in parameters.values() for v in values)]),
        "  VertexOutput result;",
        f"  result.position = {float4(position)};",
        *(f"  result.attr{k} = {float4(values)};" for k, values in parameters.items()),
        "  return result;",
        "}",
    ]
    return "\n".join(lines) + "\n"


def lower(program):
    """Return HLSL for a decoded NGG vertex program given as (byte offset, text) pairs."""
    if not program or program[-1][1] != "s_endpgm" or [line for _, line in program].count("s_endpgm") != 1:
        raise ValueError("Expected a single trailing s_endpgm")
    # The layout is a pipeline setting, so a program that packs the primitive itself is
    # checked against 16-bit indices and one that forwards v0 against pass-through input.
    for layout in LAYOUTS:
        expressions = Expressions()
        try:
            waves = [wave for shape in SUBGROUPS for wave in check_subgroup(program, shape, layout, expressions)]
            break
        except PrimitiveMismatch:
            if layout == LAYOUTS[-1]:
                raise
    # Every vertex thread of every subgroup must compute the same function of VertexID.
    positions = {value for wave in waves for value in wave.exports.get("pos0", {}).values()}
    if len(positions) != 1:
        raise ValueError("Position depends on more than VertexID")
    targets = {target for wave in waves for target in wave.exports if target.startswith("param")}
    parameters = {}
    for k in sorted(int(target[5:]) for target in targets):
        if any(wave.exports.get(f"param{k}", {}).keys() != wave.exports.get("pos0", {}).keys() for wave in waves):
            raise ValueError(f"param{k} export lanes differ from the input vertices")
        values = {value for wave in waves for value in wave.exports.get(f"param{k}", {}).values()}
        if len(values) != 1:
            raise ValueError(f"param{k} depends on more than VertexID")
        parameters[k] = values.pop()
    return emit(expressions, positions.pop(), parameters)


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
