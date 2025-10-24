"""
Local Inference Script for Bloom-QG Model.

Supports interactive mode and batch processing on MPS/CPU.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer

from bloom_qg.models import HybridModel, BLOOM_LEVELS

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Detect best available device (MPS > CPU).
    
    Returns:
        torch.device for computation
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def load_model(model_path: str, device: torch.device) -> HybridModel:
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to checkpoint file
        device: Target device
        
    Returns:
        Loaded HybridModel
    """
    logger.info("Initializing model...")
    model = HybridModel(device=str(device))
    
    if model_path:
        logger.info(f"Loading weights from {model_path}")
        model.load_trainable(model_path)
    else:
        logger.warning("No checkpoint provided, using randomly initialized weights")
    
    model.eval()
    return model


def interactive_mode(model: HybridModel):
    """
    Run interactive question generation loop.
    
    Args:
        model: Loaded HybridModel
    """
    print("\n" + "=" * 60)
    print("BLOOM-CONTROLLED QUESTION GENERATION - Interactive Mode")
    print("=" * 60)
    print(f"Available Bloom Levels: {', '.join(BLOOM_LEVELS)}")
    print("Type 'quit' to exit\n")
    
    while True:
        # Get context
        print("\n" + "-" * 60)
        context = input("Context: ").strip()
        if context.lower() == "quit":
            break
        
        # Get answer
        answer = input("Answer: ").strip()
        if answer.lower() == "quit":
            break
        
        # Get Bloom level
        bloom_level = input(f"Bloom Level ({'/'.join(BLOOM_LEVELS)}): ").strip()
        if bloom_level.lower() == "quit":
            break
        
        # Validate level
        if bloom_level not in BLOOM_LEVELS:
            print(f"❌ Invalid level! Must be one of: {', '.join(BLOOM_LEVELS)}")
            continue
        
        # Generate question
        try:
            questions = model.generate(
                contexts=[context],
                answers=[answer],
                bloom_levels=[bloom_level],
                num_beams=4,
                max_new_tokens=128
            )
            
            print("\n" + "=" * 60)
            print(f"Generated Question ({bloom_level}):")
            print("=" * 60)
            print(f"✓ {questions[0]}")
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            print(f"❌ Error: {e}")
    
    print("\nGoodbye!")


def batch_mode(model: HybridModel, test_json: str, output_path: str):
    """
    Generate questions for batch of test samples.
    
    Args:
        model: Loaded HybridModel
        test_json: Path to test JSON file
        output_path: Path to save predictions
    """
    logger.info(f"Loading test data from {test_json}")
    
    with open(test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    logger.info(f"Generating questions for {len(test_data)} samples...")
    
    predictions = []
    
    for item in test_data:
        context = item["context"]
        answer = item["answer"]
        bloom_level = item["bloom_level"]
        
        # Generate
        try:
            questions = model.generate(
                contexts=[context],
                answers=[answer],
                bloom_levels=[bloom_level],
                num_beams=4,
                max_new_tokens=128
            )
            
            predictions.append({
                "context": context,
                "answer": answer,
                "bloom_level": bloom_level,
                "reference": item.get("question", ""),
                "prediction": questions[0]
            })
            
        except Exception as e:
            logger.error(f"Failed to generate for item: {e}")
            predictions.append({
                "context": context,
                "answer": answer,
                "bloom_level": bloom_level,
                "reference": item.get("question", ""),
                "prediction": "",
                "error": str(e)
            })
    
    # Save predictions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(predictions)} predictions to {output_path}")


def default_test_examples(model: HybridModel):
    """
    Run model on default test examples.
    
    Args:
        model: Loaded HybridModel
    """
    test_cases = [
        {
            "context": "Albert Einstein developed the theory of relativity in 1905.",
            "answer": "Albert Einstein",
            "bloom_level": "Remember"
        },
        {
            "context": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "answer": "photosynthesis",
            "bloom_level": "Understand"
        },
        {
            "context": "To calculate the area of a circle, multiply pi by the radius squared.",
            "answer": "multiply pi by the radius squared",
            "bloom_level": "Apply"
        },
        {
            "context": "The main difference between mitosis and meiosis is that mitosis produces identical cells while meiosis produces genetically diverse cells.",
            "answer": "mitosis produces identical cells while meiosis produces diverse cells",
            "bloom_level": "Analyze"
        },
    ]
    
    print("\n" + "=" * 60)
    print("RUNNING DEFAULT TEST EXAMPLES")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Example {i} ({case['bloom_level']}) ---")
        print(f"Context: {case['context']}")
        print(f"Answer: {case['answer']}")
        
        questions = model.generate(
            contexts=[case["context"]],
            answers=[case["answer"]],
            bloom_levels=[case["bloom_level"]],
            num_beams=4,
            max_new_tokens=128
        )
        
        print(f"Generated: {questions[0]}")
    
    print("\n" + "=" * 60)


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(
        description="Local inference for Bloom-QG Model (MPS/CPU)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["interactive", "batch", "test"],
        help="Inference mode (default: interactive)"
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default=None,
        help="Path to test JSON for batch mode"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="predictions.json",
        help="Output path for batch predictions"
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
    
    # Get device
    device = get_device()
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Run appropriate mode
    if args.mode == "interactive":
        interactive_mode(model)
    elif args.mode == "batch":
        if not args.test_json:
            logger.error("--test_json required for batch mode")
            return
        batch_mode(model, args.test_json, args.output_path)
    elif args.mode == "test":
        default_test_examples(model)


if __name__ == "__main__":
    main()
