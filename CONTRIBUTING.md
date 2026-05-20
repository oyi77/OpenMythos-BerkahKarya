# Contributing to OpenMythos-BerkahKarya

## How to Contribute

### 1. Add Training Data
- Add new domain data to `data/finance/`
- Use JSONL format: `{"text": "..."}`
- Minimum 50 characters per sample
- Run quality check before submitting

### 2. Improve Model Architecture
- Edit modules in `open_mythos/`
- Add tests to `tests/`
- Update documentation

### 3. Add Evaluation Tasks
- Add new eval prompts to `evaluation/run_eval.py`
- Include keywords for scoring

## Data Quality Guidelines

### Good Sample
```json
{
  "text": "## Trading Analysis: XAUUSD\n\n### Setup\n- Entry: $1,950\n- Stop loss: $1,940\n- Take profit: $1,970\n- R:R: 1:2"
}
```

### Bad Sample
```json
{
  "text": "buy gold now"
}
```

### Rules
- Minimum 50 characters
- Structured with headers (##)
- Include actionable information
- Domain-specific terminology
- No spam/noise

## Pull Request Process

1. Fork the repo
2. Create feature branch
3. Add your changes
4. Run tests: `python -m pytest tests/`
5. Submit PR with description

## Code Style

- Python 3.10+
- Type hints required
- Docstrings for all functions
- Max line length: 120 chars

## License

MIT
