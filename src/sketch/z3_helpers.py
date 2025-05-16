from dataclasses import dataclass
from z3 import *
from typing import Set, List
from src.sketch.uni_parser.ast import Instruction, Register, Immediate


@dataclass
class Z3InstructionBlock:
    instructions: List[Instruction]
    input_regs: Set[str]
    output_reg: str


def bv_imm(val):
    return BitVecVal(val, 64)


def sext32(expr):
    return SignExt(32, Extract(31, 0, expr))


def zext32(expr):
    return ZeroExt(32, Extract(31, 0, expr))


def bv_mask32(expr):
    return expr & BitVecVal(0xffffffff, 64)


def resolve_operand(state, operand):
    if isinstance(operand, Register):
        return state[operand.name]
    elif isinstance(operand, Immediate):
        return BitVecVal(operand.value, 64)
    else:
        raise Exception(f"Unsupported operand type: {operand}")


z3_exec_riscv = {
    "add": lambda state, args: state.__setitem__(
        args[0].name, resolve_operand(
            state, args[1]) + resolve_operand(state, args[2])
    ),
    "addi": lambda state, args: state.__setitem__(
        args[0].name, resolve_operand(
            state, args[1]) + resolve_operand(state, args[2])
    ),
    "addiw": lambda state, args: state.__setitem__(
        args[0].name, sext32(resolve_operand(
            state, args[1]) + resolve_operand(state, args[2]))
    ),
    "slli": lambda state, args: state.__setitem__(
        args[0].name, resolve_operand(
            state, args[1]) << resolve_operand(state, args[2])
    ),
    "xor": lambda state, args: state.__setitem__(
        args[0].name, resolve_operand(
            state, args[1]) ^ resolve_operand(state, args[2])
    ),
    "sext.w": lambda state, args: state.__setitem__(
        args[0].name, sext32(resolve_operand(state, args[1]))
    ),
    "mv": lambda state, args: state.__setitem__(
        args[0].name, resolve_operand(state, args[1])
    ),
    "li": lambda state, args: state.__setitem__(
        args[0].name, resolve_operand(state, args[1])
    ),
}

z3_exec_arm64 = {
    "add": lambda state, args: state.__setitem__(
        args[0].name, resolve_operand(
            state, args[1]) + resolve_operand(state, args[2])
    ),
    "sub": lambda state, args: state.__setitem__(
        args[0].name, resolve_operand(
            state, args[1]) - resolve_operand(state, args[2])
    ),
    "subs": lambda state, args: state.__setitem__(
        args[0].name, resolve_operand(
            state, args[1]) - resolve_operand(state, args[2])
    ),
    "lsl": lambda state, args: state.__setitem__(
        args[0].name, resolve_operand(
            state, args[1]) << resolve_operand(state, args[2])
    ),
    "mov": lambda state, args: state.__setitem__(
        args[0].name, resolve_operand(state, args[1])
    ),
    "and": lambda state, args: state.__setitem__(
        args[0].name, resolve_operand(state, args[1]) & resolve_operand(state, args[2])
    ),
    "ands": lambda state, args: state.__setitem__(
        args[0].name, resolve_operand(state, args[1]) & resolve_operand(state, args[2])
    ),
}

z3_exec_x86 = {
    "addl": lambda state, args: state.__setitem__(
        args[1].name, resolve_operand(state, args[1]) + resolve_operand(state, args[0])
    ),
    "addq": lambda state, args: state.__setitem__(
        args[1].name, resolve_operand(state, args[1]) + resolve_operand(state, args[0])
    ),
    "subl": lambda state, args: state.__setitem__(
        args[1].name, resolve_operand(state, args[1]) - resolve_operand(state, args[0])
    ),
    "xorl": lambda state, args: state.__setitem__(
        args[1].name, resolve_operand(state, args[1]) ^ resolve_operand(state, args[0])
    ),
    "movl": lambda state, args: state.__setitem__(
        args[1].name, zext32(resolve_operand(state, args[0]))
    ),
    "movq": lambda state, args: state.__setitem__(
        args[1].name, resolve_operand(state, args[0])
    ),
    "movslq": lambda state, args: state.__setitem__(
        args[1].name, sext32(resolve_operand(state, args[0]))
    ),
}


def get_z3_mappings_for_lang(lang: str):
    if lang == "riscv":
        return z3_exec_riscv
    elif lang == "arm64" or lang == "arm":
        return z3_exec_arm64
    elif lang == "x86":
        return z3_exec_x86
    else:
        raise NotImplementedError(
            f"Mappings for lang {lang} are not supported yet"
        )


def extract_block_info(instructions: List[Instruction], block_lang: str) -> Z3InstructionBlock:
    defined: Set[str] = set()
    input_regs: Set[str] = set()
    output_reg: str = None

    for instr in instructions:
        if not isinstance(instr, Instruction):
            continue
        if not instr.operands:
            continue

        if block_lang == "riscv":
            dst = instr.operands[0]
            srcs = instr.operands[1:]
        elif block_lang == "arm64" or block_lang == "arm":
            dst = instr.operands[0]
            srcs = instr.operands[1:]
        elif block_lang == "x86":
            dst = instr.operands[-1]
            srcs = instr.operands[:-1]
        else:
            raise ValueError(f"Unsupported block language: {block_lang}")

        for src in srcs:
            if isinstance(src, Register) and src.name not in defined:
                input_regs.add(src.name)

        if isinstance(dst, Register):
            output_reg = dst.name
            defined.add(dst.name)

    if output_reg is None:
        raise ValueError("No output register found in block.")

    return Z3InstructionBlock(
        instructions=instructions,
        input_regs=input_regs,
        output_reg=output_reg
    )


def is_valid_block_pair(source: Z3InstructionBlock, pred: Z3InstructionBlock) -> bool:
    if len(source.input_regs) != len(pred.input_regs):
        return False

    if not source.output_reg or not pred.output_reg:
        return False

    return True


def blocks_equivalent_z3(
    source: Z3InstructionBlock,
    pred: Z3InstructionBlock,
    source_lang: str,
    pred_lang: str
) -> bool:
    if not is_valid_block_pair(source, pred):
        return False

    source_inputs = list(source.input_regs)
    pred_inputs = list(pred.input_regs)
    num_inputs = len(source_inputs)

    input_vars = [BitVec(f'in{i}', 64) for i in range(num_inputs)]
    source_state = {reg: input_vars[i] for i, reg in enumerate(source_inputs)}
    pred_state = {reg: input_vars[i] for i, reg in enumerate(pred_inputs)}

    source_exec_map = get_z3_mappings_for_lang(source_lang)
    pred_exec_map = get_z3_mappings_for_lang(pred_lang)

    # 4. Execute both blocks
    for instr in source.instructions:
        fn = source_exec_map.get(instr.name)
        if not fn:
            raise Exception(f"Unsupported source instruction: {instr.name}")
        fn(source_state, instr.operands)

    for instr in pred.instructions:
        fn = pred_exec_map.get(instr.name)
        if not fn:
            raise Exception(f"Unsupported pred instruction: {instr.name}")
        fn(pred_state, instr.operands)

    out_src = source_state[source.output_reg]
    out_pred = pred_state[pred.output_reg]

    solver = Solver()
    solver.add(out_src != out_pred)

    return solver.check() == unsat
