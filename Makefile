.PHONY: setup train run test clean

# Install pipenv dependencies
setup:
	@echo "Installing dependencies with pipenv..."
	pip install pipenv
	pipenv install --dev
	@echo "✓ Dependencies installed!"

# Train the models
train:
	@echo "Training models..."
	pipenv run python code/train.py
	@echo "✓ Training complete! Models saved to output/model/"

# Run Flask app via Docker
run:
	@echo "Building and running Docker container..."
	docker build -t whoop-coach .
	docker run -it --rm -p 9696:9696 whoop-coach
	@echo "✓ Service running on http://localhost:9696"

# Test the API endpoint
test:
	@echo "Testing API endpoint..."
	@curl -X POST http://localhost:9696/predict \
		-H "Content-Type: application/json" \
		-d @input.json
	@echo "\n✓ Test complete!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf output/model/*.bin
	rm -rf output/prediction/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleanup complete!"

# Run notebook
notebook:
	@echo "Starting Jupyter notebook..."
	pipenv run jupyter notebook notebooks/notebook.ipynb

# Full workflow: setup, train, and run
all: setup train run
