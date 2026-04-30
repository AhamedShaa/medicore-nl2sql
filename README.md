# MediCore NL2SQL

Multi-agent natural language to SQL analytics platform for the MediCore Hospital dataset.

MediCore NL2SQL lets a user ask hospital analytics questions in plain English, routes the request through a LangGraph agent workflow, generates safe read-only SQL, executes it against SQLite, and returns charts, tables, and natural language insights in a Streamlit dashboard.

## Features

- LangGraph-based multi-agent workflow for routing, SQL generation, validation, execution, repair, and interpretation.
- Read-only SQL safety checks with blocked keyword detection, schema validation, syntax validation, and retry handling.
- SQLite-backed MediCore hospital CRM dataset with dynamic schema loading.
- Streamlit dashboard with natural language querying and pre-built hospital analytics panels.
- Plotly chart rendering for bar, line, pie, and table outputs.
- Local JSON traces, with optional LangFuse observability.
- Offline unit tests for validator, fallback, state, and configuration behavior.

## Architecture Flow

The clean notebook-generated graph will be added here after the first GitHub push.

<!-- Add graph image after notebook generation, for example:

![MediCore NL2SQL architecture flow](docs/architecture-flow.png)

-->

Current pipeline:

```text
User question
    |
    v
Intent Router
    |-- blocked / unclear --> Fallback response
    |
    v
SQL Generator <----- validation retry context
    |
    v
Validator
    |-- invalid, retries left --> SQL Generator
    |-- invalid, retries exhausted --> Fallback response
    |
    v
SQL Executor
    |-- repairable execution error --> SQL Repair --> Validator
    |-- unrecoverable execution error --> Fallback response
    |
    v
Result Interpreter
    |
    v
Chart + insight + table
```

## Project Structure

```text
medicore_nl2sql/
├── agents/
│   ├── router_agent.py
│   ├── sql_generator.py
│   ├── validator_agent.py
│   ├── result_interpreter.py
│   ├── fallback_agent.py
│   └── base_agent.py
├── core/
│   ├── graph.py
│   ├── state.py
│   ├── schema_loader.py
│   ├── executor.py
│   ├── db_setup.py
│   └── tracer.py
├── dashboard/
│   ├── app.py
│   └── chart_renderer.py
├── config/
│   └── params.yaml
├── docs/
│   └── engineering_report.md
├── notebook/
│   └── medicore_exploration.ipynb
├── tests/
│   └── test_pipeline.py
├── .env.example
├── medicore_data.sql
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Clone and set up

```bash
git clone <repo-url>
cd medicore_nl2sql
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Add your OpenRouter API key to `.env`:

```text
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

LangFuse keys are optional. Without them, traces are saved locally.

### 3. Build the SQLite database

```bash
python -m core.db_setup --sql medicore_data.sql
```

This creates `medicore.db`, which is intentionally ignored by Git because it is generated from the SQL seed file.

### 4. Run the dashboard

```bash
streamlit run dashboard/app.py
```

### 5. Run the pipeline from Python

```python
from core.graph import run_pipeline

state = run_pipeline("Which doctors have the most appointments?")

if state["success"]:
    print(state["insight"])
    print(state["sql"])
else:
    print(state["fallback_message"])
```

## Dashboard

The Streamlit app has two main views:

- Natural language query view for ad hoc hospital analytics questions.
- Pre-built analytics dashboard with revenue, admission, diagnosis, appointment, payment, and doctor workload panels.

## Safety

MediCore NL2SQL is designed for read-only analytics. It blocks destructive or unsafe SQL patterns and validates generated queries before execution.

| Risk | Protection |
| --- | --- |
| Destructive SQL such as `DROP`, `DELETE`, or `UPDATE` | Blocked keyword validation |
| Unknown tables | Schema validation |
| Malformed SQL | SQL parser validation |
| Valid SQL that fails at runtime | Targeted repair loop for repairable errors |
| Ambiguous or blocked user intent | Fallback response with suggestions |

## Configuration

Runtime settings live in `config/params.yaml`, including:

- Model provider and model names.
- SQLite database path.
- Retry limits.
- Blocked SQL keywords.
- Dashboard query examples.
- LangFuse host and logging settings.

Secrets belong only in `.env`, which must not be committed.

## Tests

Run the offline test suite:

```bash
python -m pytest tests/ -v
```

Current local result:

```text
27 passed
```

The tests do not require an LLM API key.

## Notes for Reviewers

- `medicore_data.sql` is the source seed data.
- `medicore.db`, logs, local traces, caches, and video recordings are ignored by Git.
- A clearer architecture flow chart will be generated inside the notebook and added to this README after the first GitHub push.

