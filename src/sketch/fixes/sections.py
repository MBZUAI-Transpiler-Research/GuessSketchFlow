from src.sketch.sections import Section
from src.sketch.sketch import Sketch
from src.helpers.model import PredictionResult
import numpy as np
import torch


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
