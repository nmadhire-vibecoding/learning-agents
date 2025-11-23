# Udacity FNOL Claims Processing System

An AI-powered insurance claims processing system that uses Gemini 2.5 Flash for multi-stage analysis of First Notice of Loss (FNOL) reports.

## Features

The system processes insurance claims through three AI-powered stages:

### Stage I: Information Extraction
- Extracts structured information from raw FNOL text
- Identifies claim details: policyholder, vehicle, incident, damage
- Validates required fields for each claim
- Processes multiple claims in batch
- **Feedback Loop (Optional)**: LLM can review and refine its own extraction
  - Self-reviews extraction quality
  - Identifies missing or incorrect information
  - Generates refined extraction based on feedback

### Stage II: Severity Assessment & Cost Estimation
- Analyzes damage severity (Minor, Moderate, Major)
- Estimates repair costs based on damage description
- Provides reasoning for severity classification
- Generates summary statistics across all claims

### Stage III: Queue Routing & Priority Assignment
- Routes claims to appropriate queues:
  - `glass`: Minor glass-only damage
  - `fast_track`: Other minor damage
  - `material_damage`: Moderate damage
  - `total_loss`: Major damage requiring total loss handling
- Assigns priority levels (1-5, where 1 is highest priority)
- Provides queue and priority distribution statistics

## Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Gemini API key (get one for free at https://aistudio.google.com/app/apikey)

## Installation

If you don't have uv installed, install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

1. Set your Gemini API key as an environment variable:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

2. Install dependencies:

```bash
uv sync
```

## Running

Run the program with the default input file:

```bash
uv run udacity
```

Or specify a custom input file:

```bash
uv run udacity --input path/to/your/fnol.txt
# or
uv run udacity -i path/to/your/fnol.txt
```

### Enable Feedback Loop

To enable the feedback loop mechanism where the LLM reviews and refines its own Stage I extraction:

```bash
uv run udacity --feedback-loop
# or
uv run udacity -f
```

You can combine it with custom input:

```bash
uv run udacity --input path/to/your/fnol.txt --feedback-loop
```

The feedback loop process:
1. **Initial Extraction**: LLM extracts structured data from raw text
2. **Self-Review**: LLM reviews the extraction, identifies issues, and provides quality score
3. **Refinement**: LLM creates improved extraction based on feedback

This demonstrates how LLMs can iteratively improve their outputs through self-critique.

The default input file is `inputs/sample_fnol.txt` which contains 5 sample claims.

### Example Output

The system will process all claims and display:
1. **Stage I Results**: Extracted information for each claim with validation summary
2. **Stage II Results**: Severity assessment and cost estimates with summary statistics
3. **Stage III Results**: Queue assignments and priority levels with routing statistics

## Testing

Run tests using pytest:

```bash
uv run pytest
```

Or run with verbose output:

```bash
uv run pytest -v
```

Note: Tests use mocking and don't require a real API key. The test suite includes 8 tests covering all three processing stages.

## Project Structure

```
udacity/
├── src/
│   ├── __init__.py
│   ├── main.py          # Main application with 3-stage processing pipeline
│   └── models.py        # Pydantic models for all stages (FNOL, Severity, Routing)
├── tests/
│   ├── __init__.py
│   └── test_main.py     # Comprehensive tests for all stages
├── inputs/
│   └── sample_fnol.txt  # Sample FNOL input with 5 claims
├── pyproject.toml       # Project configuration and dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Architecture

The system uses a **prompt chaining** approach where each stage builds on the previous:

1. **Stage I** extracts raw claim data → `BatchFNOLInfo`
2. **Stage II** analyzes Stage I data → `BatchSeverityAssessment`
3. **Stage III** routes based on Stages I & II → `BatchQueueRouting`

Each stage uses the Gemini 2.5 Flash model with specialized prompts optimized for its specific task. All data is validated using Pydantic models to ensure type safety and data integrity.
