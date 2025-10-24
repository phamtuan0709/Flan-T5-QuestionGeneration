"""
Evaluation Script for Bloom-QG Model.

Computes BLEU, ROUGE, and METEOR metrics on predictions.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import nltk
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

logger = logging.getLogger(__name__)


def ensure_nltk_data():
    """Download required NLTK data for METEOR."""
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logger.info("Downloading NLTK wordnet data for METEOR...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Compute corpus-level BLEU score.
    
    Args:
        predictions: List of predicted questions
        references: List of reference questions
        
    Returns:
        BLEU score (0-100)
    """
    # SacreBLEU expects references as list of lists
    refs = [[ref] for ref in references]
    
    # Compute BLEU
    bleu = corpus_bleu(predictions, refs)
    return bleu.score


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    
    Args:
        predictions: List of predicted questions
        references: List of reference questions
        
    Returns:
        Dictionary with ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = defaultdict(list)
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key, value in scores.items():
            rouge_scores[key].append(value.fmeasure)
    
    # Average scores
    avg_rouge = {
        key: sum(values) / len(values)
        for key, values in rouge_scores.items()
    }
    
    return avg_rouge


def compute_meteor(predictions: List[str], references: List[str]) -> float:
    """
    Compute METEOR score.
    
    Args:
        predictions: List of predicted questions
        references: List of reference questions
        
    Returns:
        Average METEOR score (0-1)
    """
    ensure_nltk_data()
    
    meteor_scores = []
    
    for pred, ref in zip(predictions, references):
        # Tokenize
        pred_tokens = nltk.word_tokenize(pred.lower())
        ref_tokens = nltk.word_tokenize(ref.lower())
        
        # Compute METEOR
        score = meteor_score([ref_tokens], pred_tokens)
        meteor_scores.append(score)
    
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    return avg_meteor


def evaluate_by_bloom_level(
    predictions: List[Dict],
    references: List[Dict]
) -> Dict[str, Dict]:
    """
    Compute metrics per Bloom level.
    
    Args:
        predictions: List of prediction dicts with bloom_level
        references: List of reference dicts with bloom_level
        
    Returns:
        Dictionary with metrics per level
    """
    # Group by Bloom level
    level_groups = defaultdict(lambda: {"preds": [], "refs": []})
    
    for pred, ref in zip(predictions, references):
        level = pred["bloom_level"]
        level_groups[level]["preds"].append(pred["prediction"])
        level_groups[level]["refs"].append(ref["question"])
    
    # Compute metrics per level
    level_metrics = {}
    
    for level, data in level_groups.items():
        preds = data["preds"]
        refs = data["refs"]
        
        if len(preds) == 0:
            continue
        
        bleu = compute_bleu(preds, refs)
        rouge = compute_rouge(preds, refs)
        meteor = compute_meteor(preds, refs)
        
        level_metrics[level] = {
            "count": len(preds),
            "bleu": bleu,
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "meteor": meteor
        }
    
    return level_metrics


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate Bloom-QG predictions"
    )
    parser.add_argument(
        "--preds_json",
        type=str,
        required=True,
        help="Path to predictions JSON"
    )
    parser.add_argument(
        "--refs_json",
        type=str,
        default=None,
        help="Path to references JSON (optional if predictions include references)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="metrics.json",
        help="Path to save metrics JSON"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load predictions
    logger.info(f"Loading predictions from {args.preds_json}")
    with open(args.preds_json, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    
    # Load or extract references
    if args.refs_json:
        logger.info(f"Loading references from {args.refs_json}")
        with open(args.refs_json, "r", encoding="utf-8") as f:
            references = json.load(f)
    else:
        logger.info("Extracting references from predictions")
        references = predictions
    
    # Validate data
    assert len(predictions) == len(references), \
        f"Mismatch: {len(predictions)} predictions vs {len(references)} references"
    
    # Extract prediction and reference texts
    pred_texts = [item["prediction"] for item in predictions]
    ref_texts = [item.get("reference", item.get("question", "")) for item in references]
    
    # Filter out empty predictions/references
    valid_pairs = [
        (pred, ref) for pred, ref in zip(pred_texts, ref_texts)
        if pred.strip() and ref.strip()
    ]
    
    if len(valid_pairs) < len(predictions):
        logger.warning(
            f"Filtered {len(predictions) - len(valid_pairs)} empty predictions/references"
        )
    
    pred_texts, ref_texts = zip(*valid_pairs) if valid_pairs else ([], [])
    
    if not pred_texts:
        logger.error("No valid predictions/references to evaluate!")
        return
    
    logger.info(f"Evaluating {len(pred_texts)} samples...")
    
    # Compute overall metrics
    logger.info("Computing BLEU...")
    bleu_score = compute_bleu(list(pred_texts), list(ref_texts))
    
    logger.info("Computing ROUGE...")
    rouge_scores = compute_rouge(list(pred_texts), list(ref_texts))
    
    logger.info("Computing METEOR...")
    meteor = compute_meteor(list(pred_texts), list(ref_texts))
    
    # Overall metrics
    overall_metrics = {
        "bleu": bleu_score,
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "meteor": meteor,
        "num_samples": len(pred_texts)
    }
    
    # Per-level metrics
    logger.info("Computing per-level metrics...")
    level_metrics = evaluate_by_bloom_level(predictions, references)
    
    # Combine results
    results = {
        "overall": overall_metrics,
        "by_bloom_level": level_metrics
    }
    
    # Display results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Samples: {overall_metrics['num_samples']}")
    logger.info(f"BLEU: {overall_metrics['bleu']:.2f}")
    logger.info(f"ROUGE-1: {overall_metrics['rouge1']:.4f}")
    logger.info(f"ROUGE-2: {overall_metrics['rouge2']:.4f}")
    logger.info(f"ROUGE-L: {overall_metrics['rougeL']:.4f}")
    logger.info(f"METEOR: {overall_metrics['meteor']:.4f}")
    logger.info("=" * 60)
    
    # Display per-level metrics
    if level_metrics:
        logger.info("\nPer-Level Metrics:")
        logger.info("-" * 60)
        for level, metrics in level_metrics.items():
            logger.info(f"{level} (n={metrics['count']}):")
            logger.info(f"  BLEU: {metrics['bleu']:.2f}")
            logger.info(f"  ROUGE-L: {metrics['rougeL']:.4f}")
            logger.info(f"  METEOR: {metrics['meteor']:.4f}")
        logger.info("=" * 60)
    
    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
