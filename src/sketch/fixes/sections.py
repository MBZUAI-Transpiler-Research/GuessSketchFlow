import torch
import numpy as np
from typing import List

from src.sketch.sketch import Sketch
from src.sketch.sections import Section
from src.helpers.model import InferenceConfig, PredictionResult
from src.sketch.uni_parser import parse_assembly
from src.sketch.uni_parser.ast import Instruction, Symbol
from src.helpers.dataset import DatasetInstance, AssemblyLanguage
from src.sketch.instruction_mapping import get_mappings_for_lang, InstructionType


def section_confidence(section: Section, prediction: PredictionResult) -> float:
    return prediction.confidence[section.start:section.end].mean()


def remove_sections(sections: List[Section], prediction: PredictionResult) -> PredictionResult:
    pred = prediction.pred[0]
    confidence = prediction.confidence
    alignments = prediction.alignments

    sections.sort(key=lambda section: section.start)
    
    mask = np.ones(len(pred), dtype=bool)
    
    for section in sections:
        mask[section.start:section.end] = False
    
    filtered_pred = pred[mask]
    filtered_confidence = confidence[mask]
    filtered_alignments = alignments[mask] if alignments is not None else None
    
    return PredictionResult(
        instance_id=prediction.instance_id,
        source=prediction.source,
        pred=[filtered_pred],
        alignments=filtered_alignments,
        confidence=filtered_confidence
    )


def fix_duplicate_sections(sketch: Sketch, prediction: PredictionResult) -> PredictionResult:
    cleaned_pred = prediction

    sections = sketch.extract_sections(prediction.pred)
    sections_by_name = {}
    for section in sections:
        if section.name in sections_by_name:
            sections_by_name[section.name].append(section)
        else:
            sections_by_name[section.name] = [section]
    
    for section_name, sections in sections_by_name.items():
        if len(sections) > 1:
            sections.sort(key=lambda section: section_confidence(section, prediction), reverse=True)

            cleaned_pred = remove_sections(sections[1:], cleaned_pred)

    return cleaned_pred


def fix_missing_sections(
    sketch: Sketch, 
    prediction: PredictionResult,
) -> PredictionResult:
    fixed_pred = prediction

    source_lang = sketch.config.source_lang
    target_lang = sketch.config.target_lang

    pred_sections = sketch.extract_sections(prediction.pred)
    pred_sections_by_name = {section.name: section for section in pred_sections}

    source_sections = sketch.extract_sections(prediction.source)
    source_sections_by_name = {section.name: section for section in source_sections}
    
    pred_assembly = sketch.model.tokenizer.decode(prediction.pred[0])
    for line in pred_assembly.split('\n'):
        instruction = parse_assembly(line)
        if not instruction or not isinstance(instruction[0], Instruction):
            continue

        instruction = instruction[0]
        
        instruction_type, instruction_semantic = get_mappings_for_lang(target_lang).get(instruction.name, (None, None))
        
        if instruction_type != InstructionType.BRANCHING:
            continue

        if instruction_semantic not in ["branch_eq", "branch_ne", "branch_gt", "branch_ge", "branch_lt", "branch_le", "jump"]:
            continue
        
        dest_section = [operand for operand in instruction.operands if isinstance(operand, Symbol)][0].name
 
        if dest_section not in pred_sections_by_name:
            if dest_section not in source_sections_by_name:
                print("Missing section:", dest_section)
                continue
            
            source_section = source_sections_by_name[dest_section]
            source_section_assembly = sketch.model.tokenizer.decode(
                prediction.source[0][source_section.start:source_section.end]
            )
            source_section_instance = DatasetInstance(
                instance_id=source_section.instance_id,
                source=source_section_assembly,
                source_lang=AssemblyLanguage.from_str(source_lang),
                target_lang=AssemblyLanguage.from_str(target_lang),
            )

            new_section = sketch.model.predict(
                instance=source_section_instance,
                config=InferenceConfig(),
            )
            fixed_pred.pred[0] = torch.cat([
                fixed_pred.pred[0],
                new_section.pred[0],
            ])
            fixed_pred.source = torch.cat([
                fixed_pred.source,
                new_section.source,
            ])
            fixed_pred.confidence = torch.cat([
                fixed_pred.confidence,
                new_section.confidence,
            ])

            shifted_alignments = torch.tensor(
                [
                    [src_idx + source_offset for src_idx in src_idxes]
                    for src_idxes in new_section.alignments
                ]
            )
            fixed_pred.alignments = torch.cat([
                fixed_pred.alignments,
                shifted_alignments,
            ])

            # Do not generate same section twice
            pred_sections_by_name[dest_section] = Section(
                name=dest_section,
                start=-1,
                end=-1,
            )
            
    return fixed_pred
