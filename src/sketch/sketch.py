from dataclasses import dataclass

from src.sketch.pure_blocks import is_pure_line, PureInstructionBlock
from src.sketch.compare_helpers import compare_line_instructions
from typing import List, Tuple, Dict
from src.config import Config
from src.helpers.model import Model, PredictionResult
from src.sketch.uni_parser import parse_assembly
from src.sketch.z3_helpers import blocks_equivalent_z3, extract_block_info, is_valid_block_pair


@dataclass
class SketchResult:
    invalid_blocks: List[PureInstructionBlock]
    non_equivalent_blocks: List[PureInstructionBlock]
    total_blocks: List[PureInstructionBlock]


class Sketch:
    def __init__(self, config: Config, model: Model) -> None:
        self.config = config
        self.model = model

    def get_line_end_tokenized_indices(self, tokenized_sequence):
        if type(tokenized_sequence[0]) == list:
            tokenized_sequence = tokenized_sequence[0]
        decoded_seq = self.model.tokenizer.decode(
            tokenized_sequence, skip_special_tokens=False
        )
        char_to_tokenized_tok = {
            len(self.model.tokenizer.decode(tokenized_sequence[: idx + 1])): idx
            for idx in range(len(tokenized_sequence))
        }

        line_end_idxes = []
        reconstruct = ""
        seq_lines = decoded_seq.splitlines(True)

        for line in seq_lines:
            reconstruct += line
            if line[-1] == "\n":
                char_idx = len(reconstruct) - 1
            else:
                char_idx = len(reconstruct)
            while char_idx not in char_to_tokenized_tok:
                char_idx += 1
                if char_idx > len(char_to_tokenized_tok):
                    return line_end_idxes
            line_end_idxes.append(
                char_to_tokenized_tok[char_idx] + 1)  # +1 bc endidx

        return line_end_idxes

    def map_predicted_lines_to_source_lines(self, input_ids, output_ids, aligned_tokens, top_k=10, vote_threshold=1):
        input_text = self.model.tokenizer.decode(
            input_ids[0], skip_special_tokens=True
        )
        output_text = self.model.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )

        input_line_ends = self.get_line_end_tokenized_indices(input_ids[0])
        output_line_ends = self.get_line_end_tokenized_indices(output_ids[0])

        def get_line_spans(line_end_indices):
            spans = []
            start = 0
            for end in line_end_indices:
                spans.append((start, end))
                start = end
            return spans

        input_line_spans = get_line_spans(input_line_ends)
        output_line_spans = get_line_spans(output_line_ends)

        input_lines = input_text.splitlines()
        output_lines = output_text.splitlines()

        line_mapping = []

        for i, (out_start, out_end) in enumerate(output_line_spans):
            if i >= len(output_lines):
                break

            contributing_input_tokens = set()
            for j in range(out_start, out_end):
                if j < len(aligned_tokens):
                    contributing_input_tokens.update(aligned_tokens[j])

            source_line_votes = [0] * len(input_line_spans)
            for tok_idx in contributing_input_tokens:
                for k, (in_start, in_end) in enumerate(input_line_spans):
                    if in_start <= tok_idx < in_end:
                        source_line_votes[k] += 1

            top_lines = [
                idx for idx, count in sorted(
                    enumerate(source_line_votes),
                    key=lambda x: x[1],
                    reverse=True
                ) if count >= vote_threshold
            ][:top_k]

            best_score = 0.0
            best_match = None

            for idx in top_lines:
                if idx < len(input_lines):
                    score = compare_line_instructions(
                        output_lines[i],
                        input_lines[idx],
                        self.config.target_lang,
                        self.config.source_lang
                    )
                    if score > best_score:
                        best_score = score
                        best_match = idx

            if best_score > 0.0:
                line_mapping.append((i, best_match))
            else:
                line_mapping.append((i, None))

        return line_mapping

    def extract_pure_instruction_blocks(
        self, input_ids, output_ids, line_mapping
    ) -> List[PureInstructionBlock]:
        input_text = self.model.tokenizer.decode(
            input_ids[0], skip_special_tokens=True
        )
        output_text = self.model.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )

        input_lines = input_text.splitlines()
        output_lines = output_text.splitlines()

        input_line_ends = self.get_line_end_tokenized_indices(input_ids[0])
        output_line_ends = self.get_line_end_tokenized_indices(output_ids[0])

        def get_line_spans(line_end_indices):
            spans = []
            start = 0
            for end in line_end_indices:
                spans.append((start, end))
                start = end
            return spans

        input_line_spans = get_line_spans(input_line_ends)
        output_line_spans = get_line_spans(output_line_ends)

        seen_preds = set()
        blocks = []

        for i, src_idx in line_mapping:
            if i in seen_preds or src_idx is None:
                continue

            # Skip non-pure lines
            if not is_pure_line(output_lines[i], self.config.target_lang) or not is_pure_line(input_lines[src_idx], self.config.source_lang):
                continue

            # Expand up and down from current line
            pred_start, pred_end = i, i + 1
            src_start, src_end = src_idx, src_idx + 1

            seen_preds.add(pred_start)

            # Go up
            while pred_start > 0 and src_start > 0:
                score = compare_line_instructions(
                    output_lines[pred_start - 1],
                    input_lines[src_start - 1],
                    self.config.target_lang,
                    self.config.source_lang
                )

                if is_pure_line(output_lines[pred_start - 1], self.config.target_lang) and is_pure_line(input_lines[src_start - 1], self.config.source_lang) and score > 0.2:
                    pred_start -= 1
                    src_start -= 1
                    seen_preds.add(pred_start)
                else:
                    break

            # Go down
            while pred_end < len(output_lines) and src_end < len(input_lines):  
                score = compare_line_instructions(
                    output_lines[pred_end],
                    input_lines[src_end],
                    self.config.target_lang,
                    self.config.source_lang
                )

                if is_pure_line(output_lines[pred_end], self.config.target_lang) and is_pure_line(input_lines[src_end], self.config.source_lang) and score > 0.2:
                    seen_preds.add(pred_end)
                    pred_end += 1
                    src_end += 1
                else:
                    break

            # Map to token spans
            pred_token_start = output_line_spans[pred_start][0]
            pred_token_end = output_line_spans[pred_end - 1][1]

            src_token_start = input_line_spans[src_start][0]
            src_token_end = input_line_spans[src_end - 1][1]

            blocks.append(PureInstructionBlock(
                source_start=src_token_start,
                source_end=src_token_end,
                pred_start=pred_token_start,
                pred_end=pred_token_end
            ))

        return blocks

    def sketch(self, predictions: Dict[str, PredictionResult]) -> List[Tuple[PredictionResult, SketchResult]]:
        sketch_results = []
        for instance_id, pred in predictions.items():
            sketch_results.append((pred, self._sketch_single(pred)))

        return sketch_results

    def _sketch_single(self, prediction: PredictionResult) -> SketchResult:
        line_mappings = self.map_predicted_lines_to_source_lines(
            prediction.source,
            prediction.pred,
            prediction.alignments
        )

        blocks = self.extract_pure_instruction_blocks(
            prediction.source, prediction.pred, line_mappings
        )

        invalid_blocks = []
        non_equivalent_blocks = []

        for block in blocks:
            block_result = self.process_block(block, prediction)
            invalid_blocks.extend(block_result.invalid_blocks)
            non_equivalent_blocks.extend(block_result.non_equivalent_blocks)

        pred_result = SketchResult(
            invalid_blocks=invalid_blocks,
            non_equivalent_blocks=non_equivalent_blocks,
            total_blocks=blocks
        )

        return pred_result

    def process_block(self, block: PureInstructionBlock, prediction: PredictionResult) -> SketchResult:
        source_text = self.model.tokenizer.decode(
            prediction.source[0][block.source_start:block.source_end]
        ).strip()
        pred_text = self.model.tokenizer.decode(
            prediction.pred[0][block.pred_start:block.pred_end]
        ).strip()

        source_instructions = parse_assembly(source_text)
        pred_instructions = parse_assembly(pred_text)

        source_z3_block = extract_block_info(source_instructions, self.config.source_lang)
        pred_z3_block = extract_block_info(pred_instructions, self.config.target_lang)

        invalid_blocks = []
        non_equivalent_blocks = []

        if not is_valid_block_pair(source_z3_block, pred_z3_block):
            invalid_blocks.append(block)

        is_equivalent = blocks_equivalent_z3(
            source_z3_block,
            pred_z3_block,
            self.config.source_lang,
            self.config.target_lang
        )

        if not is_equivalent:
            non_equivalent_blocks.append(block)

        return SketchResult(
            invalid_blocks=invalid_blocks,
            non_equivalent_blocks=non_equivalent_blocks,
            total_blocks=[]
        )
