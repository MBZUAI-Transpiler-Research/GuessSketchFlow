
![scheme](https://github.com/user-attachments/assets/ed18a272-703f-41a3-884b-2fe5c77ed2f6)

## Sketch: Verification Pipeline for Generated Assembly

The Sketch pipeline is designed to verify the correctness of LLM-generated assembly code by comparing it against source assembly at both semantic and symbolic levels.

### Core Components

- Custom Grammar (UniAssembly):  
  We use a hand-crafted ANTLR grammar (`UniAssembly`) to parse real-world ARM64 and RISC-V assembly into a structured abstract syntax tree (AST). This allows us to reason about instructions, labels, directives, and operands in a uniform way.

---

### Processing Flow

1. Attention-Based Mapping:  
   We start with attention maps from a model that relate predicted tokens to source tokens. These are used to align each predicted line with its most likely source line.

2. Line Comparison:
   - For each aligned pair, we parse both lines using our grammar.
   - We compare the type (e.g., instruction, label, directive).
   - If both are instructions, we check semantic equivalence using a normalized instruction class (e.g., add, move, `jump`).
   - Operands are compared with:
     - Exact matching for immediates
     - Heuristic/distance-based comparison for registers

3. Line Scoring:  
   This gives us a confidence score for each predicted-to-source line match. Higher scores indicate stronger structural and semantic alignment.

---

### Pure Block Extraction

4. Finding Pure Blocks:
   - From each matched instruction line, we try to extract a pure arithmetic block.
   - A line is considered *pure* if it contains only arithmetic instructions (no memory, branching, or directives).
   - We grow the block upward and downward until we hit a non-pure instruction.

5. Block Verification (Z3):
   - For each extracted block:
     - We identify input registers (used before defined)
     - The last defined register is treated as the block’s output
   - Blocks are validated only if both predicted and source blocks have the same number of inputs.

6. Symbolic Execution in Z3:
   - Each block is converted into a symbolic function using Z3Py.
   - Registers are modeled as BitVec variables.
   - Execution is simulated using semantic mappings (e.g., add, xor, `sext.w`) translated to Z3 expressions.
   - We use z3.ForAll(...) to assert that for all possible inputs, both blocks produce identical outputs.

---

### Result

This enables us to:
- Automatically flag semantically incorrect or suspicious generated blocks
- Prepare the groundwork for automated correction, guided by Z3 counterexamples and model attention scores

The end goal is a verification-aware generation loop, where symbolic mismatch acts as a correctness signal to further train, fine-tune, or patch the model’s outputs.
