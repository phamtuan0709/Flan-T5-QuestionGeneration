# Makefile for Bloom-QG Project
# Provides convenient shortcuts for common tasks

.PHONY: help install verify prepare prepare-small train-small train-full test-run infer-test infer-interactive eval baseline clean test

help:
	@echo "Bloom-QG Makefile Commands:"
	@echo ""
	@echo "  make install           - Install all dependencies"
	@echo "  make verify            - Verify installation and dependencies"
	@echo "  make prepare           - Prepare full LearningQ dataset (~230K)"
	@echo "  make prepare-small     - Prepare small LearningQ sample (10K)"
	@echo "  make train-small       - Quick training test (10K samples, 1 epoch)"
	@echo "  make train-full        - Full training (230K samples, 3 epochs)"
	@echo "  make test-run          - Run unit tests"
	@echo "  make infer-test        - Run inference on test examples"
	@echo "  make infer-interactive - Run interactive question generation"
	@echo "  make eval              - Evaluate predictions"
	@echo "  make baseline          - Run baseline FLAN-T5"
	@echo "  make clean             - Remove generated files"
	@echo ""

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
	@echo "✅ Installation complete!"

verify:
	@echo "🔍 Verifying installation..."
	python -m bloom_qg.verify_setup

prepare:
	@echo "📚 Preparing full LearningQ dataset (~230K samples)..."
	@echo "⚠️  This may take 30-45 minutes..."
	python -m bloom_qg.data.prepare_learningq \
		--output_path data/learningq_train.json \
		--sources khan teded \
		--repo_dir ./LearningQ
	@echo "✅ Dataset preparation complete!"

prepare-small:
	@echo "📚 Preparing small LearningQ sample (10K samples)..."
	python -m bloom_qg.data.prepare_learningq \
		--output_path data/learningq_small.json \
		--sources khan \
		--limit 10000 \
		--repo_dir ./LearningQ
	@echo "✅ Small dataset ready!"

test:
	@echo "🧪 Running unit tests..."
	python -m pytest tests/ -v

train-small:
	@echo "🚀 Quick training test (10K samples, 1 epoch)..."
	python -m bloom_qg.train_gpu \
		--data_path data/learningq_small.json \
		--output_dir checkpoints_test \
		--batch_size 4 \
		--epochs 1 \
		--lr 2e-5 \
		--val_split 0.1
	@echo "✅ Test training complete!"

train-full:
	@echo "🚀 Starting full training (230K samples, 3 epochs)..."
	@echo "⏱️  Estimated time: ~20-24 hours on RTX 3090"
	python -m bloom_qg.train_gpu \
		--data_path data/learningq_train.json \
		--output_dir checkpoints \
		--batch_size 8 \
		--epochs 3 \
		--lr 2e-5 \
		--val_split 0.1

infer-test:
	@echo "Running test inference..."
	python -m bloom_qg.test_local \
		--model_path checkpoints/best_model.pt \
		--mode test

infer-interactive:
	@echo "Starting interactive mode..."
	python -m bloom_qg.test_local \
		--model_path checkpoints/best_model.pt \
		--mode interactive

infer-batch:
	@echo "Running batch inference..."
	python -m bloom_qg.test_local \
		--model_path checkpoints/best_model.pt \
		--mode batch \
		--test_json examples/test_samples.json \
		--output_path predictions.json

eval:
	@echo "Evaluating predictions..."
	python -m bloom_qg.evaluate \
		--preds_json predictions.json \
		--output_path metrics.json

baseline:
	@echo "Running baseline FLAN-T5..."
	python -m bloom_qg.baseline_flan \
		--test_json examples/test_samples.json \
		--output_path baseline_predictions.json \
		--device cuda
	@echo "Evaluating baseline..."
	python -m bloom_qg.evaluate \
		--preds_json baseline_predictions.json \
		--output_path baseline_metrics.json

clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf checkpoints/
	rm -rf checkpoints_test/
	rm -rf data/*.json
	rm -rf LearningQ/
	rm -f predictions.json
	rm -f baseline_predictions.json
	rm -f metrics.json
	rm -f baseline_metrics.json
	rm -rf __pycache__/
	rm -rf bloom_qg/__pycache__/
	rm -rf bloom_qg/*/__pycache__/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "✅ Clean complete!"

# Development helpers
lint:
	@echo "Running linter..."
	pylint bloom_qg/

format:
	@echo "Formatting code..."
	black bloom_qg/

test:
	@echo "Running tests..."
	pytest tests/ -v
