"""
Baseline FLAN-T5 Model for Comparison.

Simple prompt-based question generation without Bloom-level control
or hybrid architecture.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaselineFlanT5:
    """
    Baseline FLAN-T5 model with simple prompting.
    
    Uses prompt: "Generate a {bloom_level} level question. Context: {ctx} Answer: {ans}"
    No fusion layer, no LoRA, just basic FLAN-T5.
    """
    
    def __init__(self, model_name: str = "google/flan-t5-base", device: str = "cuda"):
        """
        Initialize baseline model.
        
        Args:
            model_name: HuggingFace model name
            device: Computation device
        """
        logger.info(f"Loading baseline FLAN-T5 from {model_name}")
        
        self.device = torch.device(device)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Baseline model loaded on {self.device}")
    
    def format_prompt(
        self,
        context: str,
        answer: str,
        bloom_level: str = "Remember"
    ) -> str:
        """
        Format input prompt for FLAN-T5.
        
        Args:
            context: Context text
            answer: Answer text
            bloom_level: Bloom taxonomy level
            
        Returns:
            Formatted prompt string
        """
        prompt = (
            f"Generate a {bloom_level} level question. "
            f"Context: {context} Answer: {answer}"
        )
        return prompt
    
    def generate(
        self,
        contexts: List[str],
        answers: List[str],
        bloom_levels: List[str] = None,
        num_beams: int = 4,
        max_new_tokens: int = 128
    ) -> List[str]:
        """
        Generate questions for batch of inputs.
        
        Args:
            contexts: List of contexts
            answers: List of answers
            bloom_levels: List of Bloom levels (optional)
            num_beams: Number of beams for beam search
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of generated questions
        """
        batch_size = len(contexts)
        
        if bloom_levels is None:
            bloom_levels = ["Remember"] * batch_size
        
        # Format prompts
        prompts = [
            self.format_prompt(ctx, ans, level)
            for ctx, ans, level in zip(contexts, answers, bloom_levels)
        ]
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode
        questions = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return questions


def run_batch_inference(
    model: BaselineFlanT5,
    test_json: str,
    output_path: str
):
    """
    Run baseline model on test data and save predictions.
    
    Args:
        model: Baseline model
        test_json: Path to test JSON
        output_path: Path to save predictions
    """
    logger.info(f"Loading test data from {test_json}")
    
    with open(test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    logger.info(f"Generating questions for {len(test_data)} samples...")
    
    predictions = []
    
    # Process in batches for efficiency
    batch_size = 8
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i:i+batch_size]
        
        contexts = [item["context"] for item in batch]
        answers = [item["answer"] for item in batch]
        bloom_levels = [item.get("bloom_level", "Remember") for item in batch]
        
        # Generate
        questions = model.generate(
            contexts=contexts,
            answers=answers,
            bloom_levels=bloom_levels
        )
        
        # Store predictions
        for item, question in zip(batch, questions):
            predictions.append({
                "context": item["context"],
                "answer": item["answer"],
                "bloom_level": item.get("bloom_level", "Remember"),
                "reference": item.get("question", ""),
                "prediction": question
            })
    
    # Save predictions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(predictions)} predictions to {output_path}")


def main():
    """Main baseline script."""
    parser = argparse.ArgumentParser(
        description="Baseline FLAN-T5 for Question Generation"
    )
    parser.add_argument(
        "--test_json",
        type=str,
        required=True,
        help="Path to test JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="baseline_predictions.json",
        help="Path to save predictions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device for inference"
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
    
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device
    
    # Initialize model
    model = BaselineFlanT5(device=device)
    
    # Run inference
    run_batch_inference(model, args.test_json, args.output_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("Baseline inference complete!")
    logger.info(f"Predictions saved to: {args.output_path}")
    logger.info("You can now evaluate with evaluate.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
