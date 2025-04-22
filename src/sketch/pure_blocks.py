from typing import List
from dataclasses import dataclass
from src.sketch.uni_parser import parse_assembly
from src.sketch.uni_parser.ast import Instruction, Register, Immediate
from src.sketch.instruction_mapping import get_mappings_for_lang, InstructionType


@dataclass
class PureInstructionBlock:
    source_start: int
    source_end: int
    pred_start: int
    pred_end: int


def is_pure_line(line: str, lang: str) -> bool:
    try:
        parsed_assembly = parse_assembly(line)

        if not parsed_assembly:
            print(f"Failed to parse line [{lang}]: {line}")
            return False

        instr = parsed_assembly[0]
        if not isinstance(instr, Instruction):
            return False

        instr_type, _ = get_mappings_for_lang(
            lang).get(instr.name, (None, None, None))
        if instr_type != InstructionType.ARITHMETIC:
            return False

        for op in instr.operands:
            if not isinstance(op, (Register, Immediate)):
                return False

        return True

    except Exception as e:
        print(f"Failed to parse line [{lang}]: {line}\nError: {str(e)}")
        return False
