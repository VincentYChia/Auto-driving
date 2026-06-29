import json
import re
from typing import Dict, Optional
import evaluate
from pycocoevalcap.cider.cider import Cider

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")

total_bleu_narration_score = 0
total_rouge_narration_score = 0
total_meteor_narration_score = 0

total_bleu_reasoning_score = 0
total_rouge_reasoning_score = 0
total_meteor_reasoning_score = 0

total_bleu_combined_score = 0
total_rouge_combined_score = 0
total_meteor_combined_score = 0

# Containers for CIDEr (Corpus-level)
narration_preds = {}
narration_refs = {}
reasoning_preds = {}
reasoning_refs = {}
combined_preds = {}
combined_refs = {}

COUNT_MISSING = False
missing_narration_count = 0
missing_reasoning_count = 0

narration_count = 0
reasoning_count = 0
combined_count = 0


def _extract_segments(text: str) -> Dict[str, str]:
    """解析单条文本，提取 Action/Narration/Reasoning/Description 等段落。"""
    if not isinstance(text, str):
        return {}
    label_pattern = re.compile(r"(Action|Narration|Reasoning|Description)\s*[:：]+", re.IGNORECASE)
    matches = list(label_pattern.finditer(text))
    if not matches:
        return {}

    segments: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        label = match.group(1).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content and label not in segments:
            segments[label] = content
    return segments


def _normalize(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    cleaned = text.strip().strip("\n")
    return cleaned or None


def _strip_coords(text: str) -> str:
    """去除文本开头的坐标部分，保留从第一个标签开始的文本。"""
    if not isinstance(text, str):
        return ""
    # 查找第一个文本标签的位置 (Narration, Reasoning, Description)
    # 忽略 Action，因为 Action 可能包含坐标
    match = re.search(r"(Narration|Reasoning|Description)\s*[:：]+", text, re.IGNORECASE)
    if match:
        return text[match.start():]
    # 如果没有找到标签，返回原文本（或者是空，取决于需求，这里假设返回原文本风险较小）
    return text


import argparse as _argparse

_parser = _argparse.ArgumentParser(
    description="Compute Nu-X NLG metrics (BLEU-4 / METEOR / ROUGE-L / CIDEr)."
)
_parser.add_argument(
    "--results", required=True,
    help="Inference results JSON: list of [pred_traj, pred_text, gt_text].",
)
_args = _parser.parse_args()

with open(_args.results, 'r') as f:
    exp = json.load(f)

for idx, item in enumerate(exp):
    # Unique ID for this sample
    sample_id = str(idx)
    
    # Fix: Handle item[1] being a string or a list
    raw_output = item[1] if len(item) > 1 else []
    if isinstance(raw_output, str):
        text_outputs = [raw_output]
    elif isinstance(raw_output, list):
        text_outputs = raw_output
    else:
        text_outputs = []

    narration: Optional[str] = None
    reasoning: Optional[str] = None

    # 先尝试逐条解析标签段落
    for text in text_outputs:
        segments = _extract_segments(text)
        if not segments:
            continue
        if narration is None:
            narration = _normalize(segments.get('narration') or segments.get('description'))
        if reasoning is None:
            reasoning = _normalize(segments.get('reasoning'))
        if narration and reasoning:
            break

    # 如果仍缺失，尝试正则兜底
    if narration is None or reasoning is None:
        for text in text_outputs:
            if narration is None:
                match = re.search(r'(Narration|Description)\s*[:：]+\s*(.+?)(?=Reasoning\s*[:：]+|$)', text, re.IGNORECASE | re.DOTALL)
                if match:
                    narration = _normalize(match.group(2))
            if reasoning is None:
                match = re.search(r'Reasoning\s*[:：]+\s*(.+)', text, re.IGNORECASE | re.DOTALL)
                if match:
                    reasoning = _normalize(match.group(1))
            if narration and reasoning:
                break

    label_raw = item[2] if len(item) > 2 else ""
    label_segments = _extract_segments(label_raw)
    narration_gt = _normalize(label_segments.get('narration') or label_segments.get('description'))
    reasoning_gt = _normalize(label_segments.get('reasoning'))

    print("Pred Narration:", narration or "(missing)")
    print("Pred Reasoning:", reasoning or "(missing)")
    print("GT Narration:", narration_gt or "(missing)")
    print("GT Reasoning:", reasoning_gt or "(missing)")
    
    if COUNT_MISSING:
        if not narration: 
            narration = ""
        if not reasoning:
            reasoning = ""
            
    # Narration Evaluation
    if narration is not None and narration_gt is not None:
        narration_bleu = bleu_metric.compute(predictions=[narration], references=[[narration_gt]])['bleu']
        narration_rouge = rouge_metric.compute(predictions=[narration], references=[narration_gt])['rougeL']
        narration_meteor = meteor_metric.compute(predictions=[narration], references=[narration_gt])['meteor']
        print(f"Narration BLEU: {narration_bleu}")
        print(f"Narration ROUGE: {narration_rouge}")
        print(f"Narration METEOR: {narration_meteor}")
        total_bleu_narration_score += narration_bleu
        total_rouge_narration_score += narration_rouge
        total_meteor_narration_score += narration_meteor
        narration_count += 1
        # CIDEr: refs/preds need list-of-strings per id
        narration_preds[sample_id] = [narration]
        narration_refs[sample_id] = [narration_gt]
    else:
        missing_narration_count += 1

    # Reasoning Evaluation
    if reasoning is not None and reasoning_gt is not None:
        reasoning_bleu = bleu_metric.compute(predictions=[reasoning], references=[[reasoning_gt]])['bleu']
        reasoning_rouge = rouge_metric.compute(predictions=[reasoning], references=[reasoning_gt])['rougeL']
        reasoning_meteor = meteor_metric.compute(predictions=[reasoning], references=[reasoning_gt])['meteor']
        print(f"Reasoning BLEU: {reasoning_bleu}")
        print(f"Reasoning ROUGE: {reasoning_rouge}")
        print(f"Reasoning METEOR: {reasoning_meteor}")
        total_bleu_reasoning_score += reasoning_bleu
        total_rouge_reasoning_score += reasoning_rouge
        total_meteor_reasoning_score += reasoning_meteor
        reasoning_count += 1
        reasoning_preds[sample_id] = [reasoning]
        reasoning_refs[sample_id] = [reasoning_gt]
    else:
        missing_reasoning_count += 1

    # Combined Evaluation (Raw Text)
    # 使用第一个 text output 作为 raw prediction (Top-1)
    pred_raw = text_outputs[0] if text_outputs else ""
    # 去除 GT 中的坐标部分，保留文本部分
    gt_raw = _strip_coords(label_raw)
    
    # 同样对 Pred 做 _strip_coords 处理，以防 Pred 也包含坐标或者杂项，
    # 并且确保对齐（都从第一个标签开始比）
    pred_combined = _strip_coords(pred_raw).strip()
    gt_combined = gt_raw.strip()
    
    if pred_combined and gt_combined:
        combined_bleu = bleu_metric.compute(predictions=[pred_combined], references=[[gt_combined]])['bleu']
        combined_rouge = rouge_metric.compute(predictions=[pred_combined], references=[gt_combined])['rougeL']
        combined_meteor = meteor_metric.compute(predictions=[pred_combined], references=[gt_combined])['meteor']
        print(f"Combined BLEU: {combined_bleu}")
        print(f"Combined ROUGE: {combined_rouge}")
        print(f"Combined METEOR: {combined_meteor}")
        total_bleu_combined_score += combined_bleu
        total_rouge_combined_score += combined_rouge
        total_meteor_combined_score += combined_meteor
        combined_count += 1
        combined_preds[sample_id] = [pred_combined]
        combined_refs[sample_id] = [gt_combined]

print("missing narration count:", missing_narration_count)
print("missing reasoning count:", missing_reasoning_count)
print("narration count:", narration_count)
print("reasoning count:", reasoning_count)
print("combined count:", combined_count)

# CIDEr Calculation
cider_scorer = Cider()

print("\n--- Computing CIDEr Scores ---")

if narration_preds:
    score, scores = cider_scorer.compute_score(narration_refs, narration_preds)
    print(f"Narration CIDEr: {score}")
else:
    print("Narration CIDEr: N/A")

if reasoning_preds:
    score, scores = cider_scorer.compute_score(reasoning_refs, reasoning_preds)
    print(f"Reasoning CIDEr: {score}")
else:
    print("Reasoning CIDEr: N/A")

if combined_preds:
    score, scores = cider_scorer.compute_score(combined_refs, combined_preds)
    print(f"Combined CIDEr: {score}")
else:
    print("Combined CIDEr: N/A")

if narration_count > 0:
    print("average bleu narration score:", total_bleu_narration_score / narration_count)
    print("average rouge narration score:", total_rouge_narration_score / narration_count)
    print("average meteor narration score:", total_meteor_narration_score / narration_count)
else:
    print("No narration samples evaluated")

if reasoning_count > 0:
    print("average bleu reasoning score:", total_bleu_reasoning_score / reasoning_count)
    print("average rouge reasoning score:", total_rouge_reasoning_score / reasoning_count)
    print("average meteor reasoning score:", total_meteor_reasoning_score / reasoning_count)
else:
    print("No reasoning samples evaluated")

if combined_count > 0:
    print("average bleu combined score:", total_bleu_combined_score / combined_count)
    print("average rouge combined score:", total_rouge_combined_score / combined_count)
    print("average meteor combined score:", total_meteor_combined_score / combined_count)
else:
    print("No combined samples evaluated")
