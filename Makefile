.PHONY: dev test test-fast test-phase1 test-phase2 test-phase3 test-phase4 test-phase5 lint format typecheck clean

# ── Installation ──────────────────────────────────────────────
dev:
	pip install -e ".[dev]"

# ── Testing ───────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -m "not slow"

test-phase1:
	pytest tests/unit/extraction/ -v --tb=short

test-phase2:
	pytest tests/unit/riemannian/ -v --tb=short

test-phase3:
	pytest tests/unit/topology/ -v --tb=short

test-phase4:
	pytest tests/unit/information/ -v --tb=short

test-phase5:
	pytest tests/unit/applications/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

# ── Code Quality ──────────────────────────────────────────────
lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/topo_llm/

# ── Cleanup ───────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf dist/ build/ htmlcov/ .coverage
