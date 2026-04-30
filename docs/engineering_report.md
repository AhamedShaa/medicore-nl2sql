# MediCore NL2SQL Platform — Engineering Report

**Course:** AI Engineer Essentials | Mini Project 04
**Module:** Multi-Agent Workflows
**System:** Multi-Agent NL2SQL Platform for MediCore Hospital

---

## 1. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interface (Streamlit)                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ Natural Language Query
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                      NL2SQL Pipeline (core/pipeline.py)          │
│                                                                  │
│  ┌─────────────────┐                                             │
│  │  Intent Router  │──→ "blocked" ──→ ⛔ Reject + message       │
│  │  (Agent 1)      │──→ "clarify" ──→ ❓ Ask for more detail    │
│  └────────┬────────┘──→ "analytics" ──▼                         │
│           │                                                      │
│  ┌────────▼────────────────────────────────┐                     │
│  │   SQL Generator Agent (Agent 2)         │                     │
│  │   + Dynamic Schema Context              │◄── DB Schema       │
│  └────────┬────────────────────────────────┘    (12 tables)     │
│           │ Generated SQL                                        │
│  ┌────────▼────────────────────────────────┐                     │
│  │   Validator Agent (Agent 3)             │                     │
│  │   Layer 1: Keyword safety               │                     │
│  │   Layer 2: Schema existence check       │                     │
│  │   Layer 3: Syntax validation            │──→ FAIL ─┐          │
│  └────────┬────────────────────────────────┘          │          │
│           │ Valid SQL                        ◄── retry (3x) ─────┘
│  ┌────────▼─────────┐                                            │
│  │   SQL Executor   │                                             │
│  │   (core module)  │──→ FAIL ──→ 🔄 Fallback Agent             │
│  └────────┬─────────┘                                            │
│           │ Raw rows                                             │
│  ┌────────▼────────────────────────────────┐                     │
│  │   Result Interpreter (Agent 4)          │                     │
│  │   - NL insight (2-3 sentences)          │                     │
│  │   - Chart spec (bar/line/pie/table)     │                     │
│  └────────┬────────────────────────────────┘                     │
│           │                                                      │
│  ┌────────▼─────────┐   ┌──────────────────┐                    │
│  │  Structured      │   │   LangFuse +      │                   │
│  │  PipelineResult  │   │   Local Traces    │                   │
│  └──────────────────┘   └──────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Query Trace Analysis

Three representative traces showing the system's behaviour in different scenarios.

---

### Trace 1 — Simple Query (Success)

**Natural Language:** "What are the top 5 revenue-generating departments?"

**Generated SQL:**
```sql
SELECT d.department_name, SUM(bi.total_amount) as revenue
FROM billing_invoices bi
JOIN departments d ON bi.department_id = d.department_id
GROUP BY d.department_name
ORDER BY revenue DESC
LIMIT 5;
```

**Validation:** PASSED — SELECT only, valid tables (`billing_invoices`, `departments`), correct syntax.

**Execution timing:**
| Agent | Latency |
|---|---|
| Intent Router | 412 ms |
| SQL Generator | 876 ms |
| Validator | 3 ms |
| SQL Executor | 19 ms |
| Result Interpreter | 1031 ms |
| **Total** | **2341 ms** |

**Output insight:** "Cardiology leads with LKR 4.2M in revenue, followed by Orthopedics at LKR 3.8M. These two departments account for over 40% of total hospital revenue."

---

### Trace 2 — Dangerous Query (Blocked)

**Natural Language:** "DELETE all patient records"

**Routing Decision:** `blocked` — The Intent Router correctly identified a destructive intent before any SQL was generated.

**Execution timing:**
| Agent | Latency |
|---|---|
| Intent Router | 398 ms |
| Fallback Agent | 0.2 ms |
| **Total** | **398 ms** |

**User Response:** "⛔ Request blocked: User is requesting a destructive DELETE operation. This system only supports read-only analytics queries."

**Key observation:** The dangerous query never reached the SQL Generator. The router acted as the first line of defence, preventing unnecessary API calls and eliminating any SQL injection risk.

---

### Trace 3 — Complex Query with Retry

**Natural Language:** "Show doctors with no-show rates above 20%"

**Attempt 1 SQL (failed):**
```sql
SELECT d.first_name, d.last_name FROM doctors d
JOIN appointment_records ar ON d.doctor_id = ar.doctor_id
-- ❌ Table 'appointment_records' doesn't exist
```

**Validation error:** "Table 'appointment_records' not found. Available tables include: appointments, doctors, patients..."

**Attempt 2 SQL (success):**
```sql
SELECT d.first_name || ' ' || d.last_name as doctor_name,
       COUNT(CASE WHEN a.status='no_show' THEN 1 END) * 100.0 / COUNT(*) as no_show_rate
FROM doctors d
JOIN appointments a ON d.doctor_id = a.doctor_id
GROUP BY d.doctor_id
HAVING no_show_rate > 20
ORDER BY no_show_rate DESC
LIMIT 100;
```

**Execution timing:**
| Agent | Attempt | Latency |
|---|---|---|
| Intent Router | — | 389 ms |
| SQL Generator | #1 | 823 ms |
| Validator | #1 FAIL | 2 ms |
| SQL Generator | #2 | 1045 ms |
| Validator | #2 PASS | 2 ms |
| SQL Executor | — | 22 ms |
| Result Interpreter | — | 1529 ms |
| **Total** | **2 attempts** | **3812 ms** |

**Key observation:** The retry mechanism added ~1 second but produced a correct result. The validator's error message was passed back to the SQL Generator as context, allowing it to fix the table name on the second attempt.

---

## 3. Reflection & Production Readiness

*(~850 words)*

### What Worked Well

The most successful design decision was **separating validation from generation**. The SQL Generator uses an LLM which is creative and non-deterministic — great for understanding intent, but unreliable for strict correctness. The Validator is purely rule-based — fast, deterministic, and testable. This separation means we can write unit tests for the Validator (we have 17 of them) without needing an API key or a live database.

The **dynamic schema loading** via `core/schema_loader.py` was also critical. On the first attempt, I tried hardcoding the 12 table names. This works until the database adds a column or table — then every prompt is stale. The dynamic approach reads the real schema at runtime, so the system self-adapts. For each table it injects the column names, types, foreign keys, and one sample row. This gives the LLM enough context to write correct JOINs without hallucinating column names.

The **three-layer validator** (keyword → schema → syntax) provides defence-in-depth. Layer 1 is a fast regex check — catches DROP, DELETE, etc. in milliseconds. Layer 2 checks table names against the real schema. Layer 3 uses `sqlparse` for syntax issues. Even if Layer 1 somehow passed a keyword, Layer 3 would catch malformed SQL. Multiple independent checks are more robust than one complex check.

### Scaling to 10,000 Queries per Day

At current design, the system makes 3 API calls per query (Router, Generator, Interpreter). At 10k queries/day that's 30,000 API calls. Key bottlenecks and mitigations:

1. **Schema loading:** Currently loads schema on every `NL2SQLPipeline()` init. At scale, cache the schema string in Redis with a 5-minute TTL. Schema rarely changes.
2. **Database connections:** Each `execute_query()` opens and closes a SQLite connection. For PostgreSQL in production, use a connection pool (e.g. `asyncpg` + `pgbouncer`). Target: pool of 20 connections for 10k daily queries.
3. **LLM latency:** Current pipeline is synchronous. At scale, the Router + Generator can run in parallel if we send the query to both simultaneously and discard the Generator's output if the Router returns "blocked". This would save ~400ms on the critical path.
4. **Rate limiting:** Add per-user rate limits (e.g. 100 queries/hour) using Redis counters. Protect against users who loop queries programmatically.

### Role-Based Access Control (RBAC)

In a real hospital deployment, not all users should see all data. Design:

- **Doctors:** Can query their own patient records only. The pipeline prepends `WHERE doctor_id = {authenticated_doctor_id}` to any query about patients.
- **Department Managers:** Can query revenue and staffing for their department only.
- **Administrators:** Full read access to all 12 tables.
- **Billing Staff:** Can query `billing_invoices` and `payments` but NOT `diagnoses` or clinical records (HIPAA compliance).

Implementation: Add a `user_context` parameter to `pipeline.run()`. The SQL Generator system prompt includes: "The authenticated user is a doctor with doctor_id=42. Only generate queries for their patients." This is simpler than post-hoc row filtering and harder to bypass.

### SQL Injection Mitigation

The current system has three protections:

1. The Router blocks queries with obvious destructive intent before SQL generation.
2. The Validator blocks any generated SQL containing `DROP`, `DELETE`, `--`, `/*` etc.
3. The Executor has a final `startswith("SELECT")` guard.

For production, add parameterised queries where the user controls filter values (e.g. "show patient 123"). Currently, patient IDs from user input are passed raw into the LLM which includes them in the SQL string. This is safe for a demo but for production: extract any user-supplied values, validate them as the expected type (integer, date string), and use SQLite's `?` parameter binding rather than string interpolation.

### Retrospective

**What I would do differently:**

- **Start with the schema loader.** I built the SQL Generator first, then realised I needed schema context. Building schema loading first would have made every subsequent step clearer.
- **Make the Validator an LLM agent.** Currently it's rule-based. An LLM validator could catch semantically invalid queries (e.g. asking for a column that doesn't make sense for the table) that regex cannot. The tradeoff is speed and cost — a fast rule check for obvious safety, then an optional LLM check for semantic correctness.
- **Multi-turn conversation memory.** The current system treats each query independently. Hospitals often want follow-up queries: "Show me the top 5 departments" → "Now break down Cardiology by doctor." A conversation buffer that appends previous results to the next prompt would make this natural. This is architecturally straightforward: pass the last 3 `PipelineResult` objects as context.
- **Better test coverage for the LLM components.** Unit tests cover the Validator well (deterministic). The Router and Generator need integration tests with recorded API responses (use `pytest-recording` or `vcr.py` to record and replay API calls without live network access).

### Conclusion

The multi-agent architecture succeeds at the core goal: hospital staff can type natural questions and receive charts and insights without writing SQL. The safety layer prevents the most common attack vectors. LangFuse provides the observability needed to debug failures in production. The most important remaining work before real deployment is RBAC, parameterised queries for user-supplied values, and a conversation memory buffer for multi-turn analytics sessions.
