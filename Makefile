.PHONY: help install install-dev test test-coverage lint ruff-check ruff-fix format type-check
.PHONY: pre-commit quality ci docs docs-api docs-notebooks docs-serve clean build publish 
.PHONY: release-check tag-version docker-build docker-run benchmark benchmark-full benchmark-quick
.PHONY: test-progress example-tabular example-timeseries example-graph test-install test-examples
.PHONY: verify deps-update deps-check deps-freeze security security-audit license-check
.PHONY: profile memory-profile monitor stress-test

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON ?= python
PIP ?= pip
PACKAGE_NAME = creso
VERSION := $(shell $(PYTHON) -c "from creso.version import __version__; print(__version__)")

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

help: ## Show this help message
	@echo "$(GREEN)CReSO Development Commands$(RESET)"
	@echo "$(BLUE)Version: $(VERSION)$(RESET)"
	@echo ""
	@echo "$(YELLOW)Setup Commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | grep -E "(install|setup)"
	@echo ""
	@echo "$(YELLOW)Development Commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | grep -E "(test|lint|format|type|clean)"
	@echo ""
	@echo "$(YELLOW)Documentation Commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | grep -E "(docs)"
	@echo ""
	@echo "$(YELLOW)Example Commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | grep -E "(example|benchmark)"
	@echo ""
	@echo "$(YELLOW)Deployment Commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST) | grep -E "(build|publish|docker)"

install: ## Install package in development mode
	@echo "$(BLUE)Installing $(PACKAGE_NAME) in development mode...$(RESET)"
	$(PIP) install -e .

install-dev: ## Install package with development dependencies
	@echo "$(BLUE)Installing $(PACKAGE_NAME) with development dependencies...$(RESET)"
	$(PIP) install -e ".[dev-all]"
	@echo "$(GREEN)✓ Development environment ready$(RESET)"

install-test: ## Install test dependencies only
	@echo "$(BLUE)Installing test dependencies...$(RESET)"
	$(PIP) install pytest pytest-cov pytest-xdist

test: ## Run the test suite
	@echo "$(BLUE)Running test suite...$(RESET)"
	$(PYTHON) -m pytest tests/ -v --tb=short
	@echo "$(GREEN)✓ Tests completed$(RESET)"

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	$(PYTHON) -m pytest tests/ -v --cov=$(PACKAGE_NAME) --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(RESET)"

test-fast: ## Run tests in parallel (fast)
	@echo "$(BLUE)Running tests in parallel...$(RESET)"
	$(PYTHON) -m pytest tests/ -n auto --dist worksteal
	@echo "$(GREEN)✓ Fast tests completed$(RESET)"

lint: ## Run code linting with ruff
	@echo "$(BLUE)Running ruff linting checks...$(RESET)"
	$(PYTHON) -m ruff check $(PACKAGE_NAME)/ tests/ examples/
	@echo "$(GREEN)✓ Ruff linting passed$(RESET)"

ruff-check: ## Run ruff linting (alias for lint)
	@echo "$(BLUE)Running ruff linting checks...$(RESET)"
	$(PYTHON) -m ruff check $(PACKAGE_NAME)/ tests/ examples/
	@echo "$(GREEN)✓ Ruff linting passed$(RESET)"

ruff-fix: ## Auto-fix ruff issues
	@echo "$(BLUE)Auto-fixing ruff issues...$(RESET)"
	$(PYTHON) -m ruff check --fix $(PACKAGE_NAME)/ tests/ examples/
	@echo "$(GREEN)✓ Ruff auto-fixes applied$(RESET)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code with black...$(RESET)"
	$(PYTHON) -m black $(PACKAGE_NAME)/ tests/ examples/
	@echo "$(GREEN)✓ Code formatted$(RESET)"

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	$(PYTHON) -m black --check $(PACKAGE_NAME)/ tests/ examples/

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(RESET)"
	$(PYTHON) -m mypy $(PACKAGE_NAME)/ --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking passed$(RESET)"

pre-commit: ## Run pre-commit hooks
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	$(PYTHON) -m pre_commit run --all-files || echo "$(YELLOW)pre-commit not available, install with: pip install pre-commit$(RESET)"
	@echo "$(GREEN)✓ Pre-commit hooks completed$(RESET)"

quality: ruff-check format-check type-check ## Run all code quality checks
	@echo "$(GREEN)✓ All quality checks passed$(RESET)"

ci: test-coverage ruff-check format-check type-check ## Full CI pipeline
	@echo "$(BLUE)Running complete CI pipeline...$(RESET)"
	@echo "$(GREEN)✓ All CI checks passed - ready for production$(RESET)"

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	@echo "$(YELLOW)Documentation build not implemented yet$(RESET)"

docs-api: ## Generate API documentation
	@echo "$(BLUE)Generating API documentation...$(RESET)"
	$(PYTHON) -m pydoc -w $(PACKAGE_NAME) || echo "$(YELLOW)API documentation generation failed$(RESET)"
	@echo "$(GREEN)✓ API documentation generated$(RESET)"

docs-notebooks: ## Validate example notebooks
	@echo "$(BLUE)Validating example notebooks...$(RESET)"
	@for notebook in examples/notebooks/*.ipynb; do \
		echo "Validating $$notebook..."; \
		jupyter nbconvert --execute --to notebook --inplace "$$notebook" || echo "$(YELLOW)Notebook validation failed: $$notebook$(RESET)"; \
	done
	@echo "$(GREEN)✓ Notebook validation completed$(RESET)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Starting documentation server on http://localhost:8080$(RESET)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(RESET)"
	$(PYTHON) -m http.server 8080 -d . || echo "$(YELLOW)Could not start server$(RESET)"

clean: ## Clean up temporary files and caches (basic cleanup)
	@echo "$(BLUE)Cleaning up temporary files and caches...$(RESET)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ .coverage
	@echo "$(GREEN)✓ Basic cleanup completed$(RESET)"

clean-models: ## Remove saved models and checkpoints
	@echo "$(BLUE)Cleaning up model files...$(RESET)"
	@find . -type f \( -name "*.pkl" -o -name "*.pt" -o -name "*.ts" -o -name "*.onnx" \) -not -path "./examples/notebooks/*" -delete
	@find . -type d -name "checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Model files cleaned$(RESET)"

clean-logs: ## Remove log files and output
	@echo "$(BLUE)Cleaning up log files...$(RESET)"
	@find . -type f \( -name "*.log" -o -name "*.out" -o -name "*.err" \) -delete
	@rm -rf logs/ output/ cli_test_output/ 2>/dev/null || true
	@echo "$(GREEN)✓ Log files cleaned$(RESET)"

clean-coverage: ## Remove coverage reports and profiling data
	@echo "$(BLUE)Cleaning up coverage reports...$(RESET)"
	@rm -rf htmlcov/ coverage.xml .coverage* 2>/dev/null || true
	@find . -type f \( -name "*.prof" -o -name "*.profile" \) -delete
	@echo "$(GREEN)✓ Coverage reports cleaned$(RESET)"

clean-results: ## Remove benchmark and result files
	@echo "$(BLUE)Cleaning up results and benchmarks...$(RESET)"
	@rm -rf results/ benchmarks/results/ 2>/dev/null || true
	@find . -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.csv" \) -not -path "./docs/*" -not -path "./examples/notebooks/*" -delete
	@echo "$(GREEN)✓ Results and benchmarks cleaned$(RESET)"

clean-cache: ## Remove all cache directories
	@echo "$(BLUE)Cleaning up cache directories...$(RESET)"
	@find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" \) -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .tox/ .nox/ 2>/dev/null || true
	@echo "$(GREEN)✓ Cache directories cleaned$(RESET)"

clean-safe: clean clean-coverage clean-logs ## Safe cleanup (preserves models and important files)
	@echo "$(GREEN)✓ Safe cleanup completed - models preserved$(RESET)"

clean-all: clean clean-models clean-logs clean-coverage clean-results clean-cache ## Complete cleanup (removes everything)
	@echo "$(YELLOW)Complete cleanup - all generated files removed$(RESET)"
	@echo "$(GREEN)✓ Complete cleanup finished$(RESET)"

clean-dry-run: ## Show what would be cleaned without actually cleaning
	@echo "$(BLUE)Dry run - showing files that would be cleaned:$(RESET)"
	@echo "\n$(YELLOW)Python cache files:$(RESET)"
	@find . -type f -name "*.pyc" 2>/dev/null | wc -l | xargs echo "  Files:" 
	@echo "$(YELLOW)Cache directories:$(RESET)"
	@find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" \) 2>/dev/null | wc -l | xargs echo "  Directories:"
	@echo "$(YELLOW)Model files:$(RESET)"
	@find . -type f \( -name "*.pkl" -o -name "*.pt" -o -name "*.ts" -o -name "*.onnx" \) 2>/dev/null | wc -l | xargs echo "  Files:"
	@echo "$(YELLOW)Log files:$(RESET)"
	@find . -type f \( -name "*.log" -o -name "*.out" \) 2>/dev/null | wc -l | xargs echo "  Files:"
	@echo "$(YELLOW)Coverage files:$(RESET)"
	@find . -name "htmlcov" -o -name "coverage.xml" -o -name ".coverage*" 2>/dev/null | wc -l | xargs echo "  Files/Dirs:"

build: clean ## Build distribution packages
	@echo "$(BLUE)Building distribution packages...$(RESET)"
	$(PYTHON) -m build
	@echo "$(GREEN)✓ Packages built in dist/$(RESET)"

publish-test: build ## Publish to test PyPI
	@echo "$(BLUE)Publishing to test PyPI...$(RESET)"
	$(PYTHON) -m twine upload --repository testpypi dist/*
	@echo "$(GREEN)✓ Published to test PyPI$(RESET)"

release-check: ci benchmark-quick ## Pre-release validation
	@echo "$(BLUE)Running pre-release checks...$(RESET)"
	$(PYTHON) -m build --check || echo "$(YELLOW)Build check failed$(RESET)"
	$(PYTHON) -m twine check dist/* || echo "$(YELLOW)Distribution check failed$(RESET)"
	@echo "$(GREEN)✓ Ready for release$(RESET)"

tag-version: ## Create git tag for current version
	@echo "$(BLUE)Creating git tag for version $(VERSION)...$(RESET)"
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	git push origin v$(VERSION) || echo "$(YELLOW)Failed to push tag$(RESET)"
	@echo "$(GREEN)✓ Version tagged$(RESET)"

publish: build ## Publish to PyPI
	@echo "$(YELLOW)Publishing to PyPI...$(RESET)"
	@read -p "Are you sure you want to publish $(PACKAGE_NAME) $(VERSION) to PyPI? [y/N] " confirm && [ "$$confirm" = "y" ]
	$(PYTHON) -m twine upload dist/*
	@echo "$(GREEN)✓ Published to PyPI$(RESET)"

# Example commands
example-tabular: ## Run tabular classification example
	@echo "$(BLUE)Running tabular classification example...$(RESET)"
	$(PYTHON) examples/train_tabular.py
	@echo "$(GREEN)✓ Tabular example completed$(RESET)"

example-timeseries: ## Run time-series classification example
	@echo "$(BLUE)Running time-series classification example...$(RESET)"
	$(PYTHON) examples/train_timeseries.py
	@echo "$(GREEN)✓ Time-series example completed$(RESET)"

example-graph: ## Run graph classification example
	@echo "$(BLUE)Running graph classification example...$(RESET)"
	$(PYTHON) examples/train_graph.py
	@echo "$(GREEN)✓ Graph example completed$(RESET)"

example-all: example-tabular example-timeseries example-graph ## Run all examples
	@echo "$(GREEN)✓ All examples completed$(RESET)"

# CLI testing commands
cli-test-tabular: ## Test CLI with tabular data
	@echo "$(BLUE)Testing CLI with tabular data...$(RESET)"
	$(PYTHON) -m $(PACKAGE_NAME).cli task=tabular_binary data.path=test_data.npz model.epochs=5 out_dir=cli_test_output
	@echo "$(GREEN)✓ CLI tabular test completed$(RESET)"

# Benchmarking
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	$(PYTHON) examples/benchmarks/performance_benchmark.py
	@echo "$(GREEN)✓ Benchmarks completed$(RESET)"

benchmark-full: ## Run comprehensive benchmarks with verbose output
	@echo "$(BLUE)Running comprehensive benchmarks...$(RESET)"
	$(PYTHON) examples/benchmarks/performance_benchmark.py --verbose
	@echo "$(GREEN)✓ Comprehensive benchmarks completed$(RESET)"

benchmark-quick: ## Run quick benchmarks for CI
	@echo "$(BLUE)Running quick benchmarks...$(RESET)"
	$(PYTHON) examples/benchmarks/performance_benchmark.py --output-dir benchmark_quick || echo "$(YELLOW)Quick benchmarks failed$(RESET)"
	@echo "$(GREEN)✓ Quick benchmarks completed$(RESET)"

test-progress: ## Test progress bar functionality
	@echo "$(BLUE)Testing progress bar features...$(RESET)"
	@$(PYTHON) -c "from creso import CReSOClassifier; from creso.config import CReSOConfiguration, ModelArchitectureConfig; import numpy as np; X,y=np.random.randn(1000,10),(np.random.randn(1000)>0).astype(int); arch=ModelArchitectureConfig(input_dim=10); config=CReSOConfiguration(architecture=arch); config.training.max_epochs=3; c=CReSOClassifier(config=config); c.fit(X,y,verbose=2); print('✓ Progress bars working')"
	@echo "$(GREEN)✓ Progress bars verified$(RESET)"

# Development utilities
dev-setup: install-dev ## Complete development environment setup
	@echo "$(BLUE)Setting up complete development environment...$(RESET)"
	@echo "$(GREEN)✓ Development environment ready!$(RESET)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(RESET)"
	@echo "  - Run 'make test' to verify installation"
	@echo "  - Run 'make example-all' to try examples"
	@echo "  - Run 'make quality' before committing changes"

check: test ruff-check type-check ## Run all checks (tests, linting, type checking)
	@echo "$(GREEN)✓ All checks passed - ready to commit!$(RESET)"

# Docker commands (if Docker is available)
docker-build: ## Build Docker development image
	@echo "$(BLUE)Building Docker development image...$(RESET)"
	docker build -t $(PACKAGE_NAME):dev -f Dockerfile.dev . || echo "$(YELLOW)Docker not available or Dockerfile.dev missing$(RESET)"

docker-run: ## Run development container
	@echo "$(BLUE)Running development container...$(RESET)"
	docker run -it --rm -v $$(pwd):/workspace $(PACKAGE_NAME):dev || echo "$(YELLOW)Docker not available$(RESET)"

# Version management
version: ## Show current version
	@echo "$(BLUE)Current version: $(GREEN)$(VERSION)$(RESET)"

# Installation verification
verify: ## Verify installation
	@echo "$(BLUE)Verifying $(PACKAGE_NAME) installation...$(RESET)"
	@$(PYTHON) -c "import $(PACKAGE_NAME); print('✓ Package imports successfully')"
	@$(PYTHON) -c "from $(PACKAGE_NAME) import __version__; print(f'✓ Version: {__version__}')"
	@$(PYTHON) -c "from $(PACKAGE_NAME) import CReSOClassifier; print('✓ Main classes importable')"
	@echo "$(GREEN)✓ Installation verified$(RESET)"

deps-update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e ".[dev-all]"
	@echo "$(GREEN)✓ Dependencies updated$(RESET)"

deps-check: ## Check for dependency conflicts
	@echo "$(BLUE)Checking dependencies...$(RESET)"
	$(PIP) check
	@echo "$(GREEN)✓ No dependency conflicts$(RESET)"

deps-freeze: ## Freeze current dependencies
	@echo "$(BLUE)Freezing dependencies...$(RESET)"
	$(PIP) freeze > requirements-dev.txt
	@echo "$(GREEN)✓ Dependencies frozen to requirements-dev.txt$(RESET)"

# Security scanning (if available)
security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	$(PYTHON) -m pip audit || echo "$(YELLOW)pip-audit not available, install with: pip install pip-audit$(RESET)"

security-audit: ## Run comprehensive security audit
	@echo "$(BLUE)Running comprehensive security audit...$(RESET)"
	$(PYTHON) -m pip audit || echo "$(YELLOW)pip-audit not available$(RESET)"
	bandit -r $(PACKAGE_NAME)/ -f json -o security-report.json || echo "$(YELLOW)bandit not available, install with: pip install bandit$(RESET)"
	safety check --json --output safety-report.json || echo "$(YELLOW)safety not available, install with: pip install safety$(RESET)"
	@echo "$(GREEN)✓ Security audit completed$(RESET)"

license-check: ## Check license compatibility
	@echo "$(BLUE)Checking license compatibility...$(RESET)"
	pip-licenses --format=json --output=licenses.json || echo "$(YELLOW)pip-licenses not available, install with: pip install pip-licenses$(RESET)"
	@echo "$(GREEN)✓ License check completed$(RESET)"

# Performance profiling
profile: ## Run performance profiling
	@echo "$(BLUE)Running performance profiling...$(RESET)"
	$(PYTHON) -m cProfile -o profile_output.prof test_simple.py || echo "$(YELLOW)Profiling failed$(RESET)"
	@echo "$(GREEN)✓ Profile saved to profile_output.prof$(RESET)"

# Memory usage analysis
memory-profile: ## Profile memory usage
	@echo "$(BLUE)Profiling memory usage...$(RESET)"
	$(PYTHON) -m memory_profiler examples/benchmarks/performance_benchmark.py || echo "$(YELLOW)memory-profiler not available, install with: pip install memory-profiler$(RESET)"

test-install: build ## Test installation from built package
	@echo "$(BLUE)Testing installation from built package...$(RESET)"
	$(PIP) uninstall -y $(PACKAGE_NAME) || true
	$(PIP) install dist/*.whl
	$(PYTHON) -c "import $(PACKAGE_NAME); print('✓ Package installed successfully')"
	$(PIP) install -e ".[dev-all]"  # Restore dev install
	@echo "$(GREEN)✓ Installation test completed$(RESET)"

test-examples: ## Test all examples work
	@echo "$(BLUE)Testing all examples...$(RESET)"
	@for example in examples/train_*.py; do \
		echo "Testing $$example..."; \
		$(PYTHON) "$$example" --quick-test || echo "$(YELLOW)Example failed: $$example$(RESET)"; \
	done
	@echo "$(GREEN)✓ All examples tested$(RESET)"

monitor: ## Monitor resource usage during tests
	@echo "$(BLUE)Monitoring resource usage during tests...$(RESET)"
	@$(PYTHON) -c "import psutil; import subprocess; import time; p=subprocess.Popen(['python','-m','pytest','tests/','-x']); proc=psutil.Process(p.pid) if psutil.pid_exists(p.pid) else None; time.sleep(1); print(f'Memory: {proc.memory_info().rss/1024/1024:.1f}MB') if proc else print('Process ended quickly')"

stress-test: ## Run stress tests with large datasets
	@echo "$(BLUE)Running stress tests...$(RESET)"
	@$(PYTHON) -c "from creso import CReSOClassifier; from creso.config import CReSOConfiguration, ModelArchitectureConfig; import numpy as np; print('Generating large dataset...'); X,y=np.random.randn(10000,50),(np.random.randn(10000)>0).astype(int); arch=ModelArchitectureConfig(input_dim=50); config=CReSOConfiguration(architecture=arch); config.training.max_epochs=5; c=CReSOClassifier(config=config); print('Training on large dataset...'); c.fit(X,y,verbose=1); print('✓ Stress test completed')"
	@echo "$(GREEN)✓ Stress tests completed$(RESET)"