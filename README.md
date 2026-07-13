# LLM Prompting Design & Evaluation Framework

This repository is a modular, lightweight, and structured template designed for prompt engineering research, LLM inference, and pipeline evaluation. It is structured for **inference-only** (no training) workloads, supporting multiple API-based LLMs like Gemini (using the Google GenAI SDK) and OpenAI.

---

## 📂 Directory Layout

```text
acm2026/
├── configs/                  # Configurations (YAML) for models and pipeline parameters
├── data/                     # Local data storage (inputs, logs, goldens)
├── notebooks/                # Jupyter Notebooks for prompt tuning and visualization
├── prompts/                  # Prompt templates & system instructions (kept as separate text files)
├── src/                      # Project source code modules
│   ├── evaluation/           # Evaluation metrics and logic
│   ├── llm/                  # Wrappers/clients for LLM APIs (Gemini, OpenAI, etc.)
│   ├── methods/              # Implementation of prompting strategies (Base, CoT, etc.)
│   ├── prompts/              # Prompt file loaders
│   └── utils/                # Logging, IO helpers, and common utilities
├── tests/                    # Unit & integration testing using pytest
├── .env.example              # Key template for API credentials
├── .gitignore                # Files excluded from version control
├── main.py                   # CLI entrypoint to execute inference & evaluation
└── requirements.txt          # Project dependencies
```

---

## 🛠️ Setup Instructions

### 1. Prerequisites
Ensure you have **Python 3.10+** installed. 

### 2. Installation
Clone the repository and install dependencies:
```bash
# Navigate to the project folder
cd acm2026

# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install required libraries
pip install -r requirements.txt
```

### 3. API Keys Configuration
Copy `.env.example` to `.env` and fill in your API credentials:
```bash
cp .env.example .env
```
Edit the `.env` file:
```ini
GEMINI_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key
```

---

## 🚀 Usage

### Running Inferences
To run a prompting pipeline on sample data using the command-line interface:
```bash
python main.py --config configs/default_config.yaml --method cot
```

### Running Tests
To ensure the prompt loader and LLM interface work correctly, run:
```bash
pytest
```

---

## 💡 How to Extend

### Add a New Prompt Template
1. Create a `.txt` file inside `prompts/templates/` (e.g., `prompts/templates/few_shot_template.txt`).
2. Use curly braces `{}` for variables you want to inject at runtime (e.g. `{context}`, `{question}`).

### Add a New Prompting Method
1. Create a new file in `src/methods/` (e.g., `src/methods/react.py`).
2. Inherit from the base class `BasePromptingMethod` in [src/methods/base.py](file:///d:/Study/NCKH/ACM2026/acm2026/src/methods/base.py).
3. Implement the `run` method to invoke your custom prompting control-flow.
