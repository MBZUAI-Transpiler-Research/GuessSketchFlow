from enum import Enum, auto


class InstructionType(Enum):
    ARITHMETIC = auto()
    MEMORY = auto()
    BRANCHING = auto()
    SYSTEM = auto()  # optional: for things like `nop`, `call`


riscv_mapping = {
    "add":     (InstructionType.ARITHMETIC, "add"),
    "addi":    (InstructionType.ARITHMETIC, "add"),
    "addiw":   (InstructionType.ARITHMETIC, "add"),
    "slli":    (InstructionType.ARITHMETIC, "shift"),
    "xor":     (InstructionType.ARITHMETIC, "xor"),
    "sext.w":  (InstructionType.ARITHMETIC, "sign_extend"),
    "mv":      (InstructionType.ARITHMETIC, "move"),
    "li":      (InstructionType.ARITHMETIC, "move"),

    "nop":     (InstructionType.SYSTEM, "nop"),

    "ld":      (InstructionType.MEMORY, "load"),
    "lw":      (InstructionType.MEMORY, "load"),
    "lbu":     (InstructionType.MEMORY, "load"),
    "sd":      (InstructionType.MEMORY, "store"),
    "sw":      (InstructionType.MEMORY, "store"),
    "sb":      (InstructionType.MEMORY, "store"),
    "lla":     (InstructionType.MEMORY, "load_address"),
    "la":      (InstructionType.MEMORY, "load_address"),

    "j":       (InstructionType.BRANCHING, "jump"),
    "jr":      (InstructionType.BRANCHING, "return"),
    "call":    (InstructionType.BRANCHING, "call"),
    "beq":     (InstructionType.BRANCHING, "branch_eq"),
    "bne":     (InstructionType.BRANCHING, "branch_ne"),
    "bgt":     (InstructionType.BRANCHING, "branch_gt"),
}

arm64_mapping = {
    "add":     (InstructionType.ARITHMETIC, "add"),
    "subs":    (InstructionType.ARITHMETIC, "subtract"),
    "sub":     (InstructionType.ARITHMETIC, "subtract"),
    "lsl":     (InstructionType.ARITHMETIC, "shift"),
    "mov":     (InstructionType.ARITHMETIC, "move"),

    "nop":     (InstructionType.SYSTEM, "nop"),

    "ldr":     (InstructionType.MEMORY, "load"),
    "ldrb":    (InstructionType.MEMORY, "load"),
    "ldrsw":   (InstructionType.MEMORY, "load"),
    "str":     (InstructionType.MEMORY, "store"),
    "strb":    (InstructionType.MEMORY, "store"),
    "ldp":     (InstructionType.MEMORY, "load"),
    "stp":     (InstructionType.MEMORY, "store"),
    "adrp":    (InstructionType.MEMORY, "load_address"),

    "cmp":     (InstructionType.BRANCHING, "compare"),
    "cmn":     (InstructionType.BRANCHING, "compare_neg"),
    "b":       (InstructionType.BRANCHING, "jump"),
    "bl":      (InstructionType.BRANCHING, "call"),
    "ret":     (InstructionType.BRANCHING, "return"),
    "beq":     (InstructionType.BRANCHING, "branch_eq"),
    "bne":     (InstructionType.BRANCHING, "branch_ne"),
    "bgt":     (InstructionType.BRANCHING, "branch_gt"),
}


def get_mappings_for_lang(lang: str):
    if lang == "riscv":
        return riscv_mapping
    elif lang == "arm64":
        return arm64_mapping
    else:
        raise NotImplementedError(
            f"Mappings for lang {lang} are not supported yet"
        )
