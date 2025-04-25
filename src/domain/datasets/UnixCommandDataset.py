"""Dataset implementation for Unix command assembly code."""

import json
from pathlib import Path
import subprocess
from typing import Dict, List, Union
import re
from src.helpers.dataset import Dataset, DatasetInstance, AssemblyLanguage, DatasetConfig


@Dataset.register('unix_commands')
class UnixCommandDataset(Dataset):
    """Dataset for Unix command assembly code.

    This dataset expects data in a specific directory structure:
    data/
    ├──datasets/
    │   ├── UnixCommands/
    │       ├── example.c/  # Dataset C file entry
    ├──processed/
        ├── ARM64/      # ARM64 assembly files
        ├── X86/        # x86 assembly files
        ├── RISCV/      # RISC-V assembly files
        ├── MIPS/      # MIPS assembly files
    """

    # Map architecture to file suffix
    arch_suffix = {
        'x86': 'x86',
        'arm64': 'arm',
        'riscv': 'risc'
    }

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.dataset_name = 'UnixCommands'
        self.source_arch = config.source_lang
        self.target_arch = config.target_lang

        # Validate architectures
        valid_archs = {'x86', 'arm64', 'riscv'}
        if self.source_arch not in valid_archs or self.target_arch not in valid_archs:
            raise ValueError(
                f"Invalid architecture. Must be one of: {valid_archs}")

        # Load dataset metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load dataset metadata from source files and JSONL."""
        # Convert dataset name to Path and construct paths
        base_path = Path(self.dataset_name)

        # Load source C files
        source_dir = base_path / 'datasets' / self.dataset_name
        c_files = list(source_dir.glob('*.c'))

        # Load assembly files
        source_jsonl = Path('data/processed') / self.source_arch.upper() / \
            f'UnixCommands_{self.arch_suffix[self.source_arch]}.jsonl'
        target_jsonl = Path('data/processed') / self.target_arch.upper() / \
            f'UnixCommands_{self.arch_suffix[self.target_arch]}.jsonl'
        print(f"Source JSONL: {source_jsonl}")
        print(f"Target JSONL: {target_jsonl}")
        if not source_jsonl.exists() or not target_jsonl.exists():
            raise ValueError(f"Missing JSONL files for {self.dataset_name}")

        metadata = {
            'c_files': {f.stem: f.read_text() for f in c_files},
            'source': self._load_jsonl(source_jsonl),
            'target': self._load_jsonl(target_jsonl)
        }

        return metadata

    def _load_jsonl(self, file_path: Path, log_file: bool = False) -> List[Dict]:
        """Load entries from a JSONL file.

        Args:
            file_path: Path to the JSONL file
            log_file: Whether to log the file being loaded

        Returns:
            List of entries from the JSONL file
        """
        entries = []
        if log_file:
            self.logger.info(f"Loading JSONL file: {file_path}")
        print(f"Loading file: {file_path}")
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    entries.append(entry)
        print(f"Loaded {len(entries)} entries from {file_path}")
        return entries

    def load_data(self) -> List[DatasetInstance]:
        """Load all instances for inference."""
        instances = []

        # Create instances from all available data
        print(f"Source entries: {len(self.metadata['source'])}")
        print(f"Target entries: {len(self.metadata['target'])}")
        for idx in range(len(self.metadata['source'])):
            source_entry = self.metadata['source'][idx]
            target_entry = self.metadata['target'][idx]

            # Get C source code if available
            c_source = self.metadata['c_files'].get(
                source_entry['source'].replace('.s', ''), '')

            arch_map = {'x86': 'x86', 'arm64': 'arm', 'riscv': 'risc'}
            source_key = arch_map[self.source_arch]
            target_key = arch_map[self.target_arch]

            print(
                f"Creating instance {idx} with source key {source_key} and target key {target_key}"
            )
            print(f"Source entry keys: {source_entry.keys()}")
            print(f"Target entry keys: {target_entry.keys()}")

            instance = DatasetInstance(
                instance_id=f"{self.dataset_name}/{source_entry['source']}",
                source_lang=AssemblyLanguage(self.source_arch),
                target_lang=AssemblyLanguage(self.target_arch),
                source=source_entry[source_key],  # Contains assembly code
                target=target_entry[target_key],  # Contains assembly code
                metadata={
                    'source_file': source_entry['source'],
                    'c_source': c_source,
                    'source_output': source_entry.get(f'{source_key}_output', ''),
                    'target_output': target_entry.get(f'{target_key}_output', ''),
                }
            )
            instances.append(instance)

        print(f"Created {len(instances)} instances")
        self.logger.info(f"Loaded {len(instances)} instances")
        self._instances = instances
        return instances

    def evaluate(self, predictions: Dict[str, Dict]) -> Dict[str, Union[float, Dict[str, str]]]:
        """Evaluate transpilation quality by checking if assembly code can be compiled.

        Returns:
            Dict containing:
            - compile_rate: Overall compilation success rate
            - per_command_results: Dict mapping command names to compilation results
            - errors: Dict mapping command names to error stacktraces
        """
        metrics = {}
        errors = {}
        per_command_results = {}

        total = len(predictions)

        compile_success = 0

        for instance_id, pred in predictions.items():
            instance = [
                i for i in self._instances if i.instance_id == instance_id][0]
            instance_id = instance.instance_id.split('/')[-1]

            # Create temporary directory for compilation
            temp_dir = Path(f"/tmp/UnixCommands")
            temp_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Write prediction to assembly file
                asm_file = temp_dir / f"{instance_id}.s"
                raw_asm = pred.pred_dec
                match = re.search(r"```(?:armasm)?\s*(.*?)```", raw_asm, re.DOTALL)
                if match:
                    stripped_asm = match.group(1).strip()
                    print(f"[DEBUG] Regex STRIPPED ASM repr:\n{repr(stripped_asm)}")
                else:
                    stripped_asm = raw_asm.strip()
                    print(f"[DEBUG] Fallback STRIPPED ASM repr:\n{repr(stripped_asm)}")

                cleaned_asm = stripped_asm.replace("\r\n", "\n").replace("\r", "\n")
                asm_file.write_text(cleaned_asm)

                # Try to compile the assembly
                if self.target_arch == 'arm64':
                    compile_cmd = f"as -arch arm64 {asm_file} -o {temp_dir}/out.o && "
                    compile_cmd += f"ld -arch arm64 {temp_dir}/out.o -o {temp_dir}/out"
                elif self.target_arch == 'x86':
                    compile_cmd = f"as {asm_file} -o {temp_dir}/out.o && "
                    compile_cmd += f"ld {temp_dir}/out.o -o {temp_dir}/out"
                else:  # riscv
                    compile_cmd = f"riscv64-unknown-elf-as {asm_file} -o {temp_dir}/out.o && "
                    compile_cmd += f"riscv64-unknown-elf-ld {temp_dir}/out.o -o {temp_dir}/out"

                # Run compilation
                result = subprocess.run(
                    compile_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=temp_dir
                )

                print(result)

                if result.returncode == 0:
                    compile_success += 1
                    per_command_results[instance_id] = "success"
                else:
                    per_command_results[instance_id] = "failed"
                    errors[instance_id] = result.stderr

            except Exception as e:
                import traceback
                per_command_results[instance_id] = "error"
                errors[instance_id] = traceback.format_exc()
                self.logger.error(f"Error compiling {instance_id}: {e}")

            finally:
                # Cleanup
                import shutil
                shutil.rmtree(temp_dir)

        # Calculate overall metrics
        metrics['compile_rate'] = compile_success / total if total > 0 else 0
        metrics['per_command_results'] = per_command_results
        metrics['errors'] = errors

        return metrics
