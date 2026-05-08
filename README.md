# LAB | NormalObjects - Creative Complaint Handler (LangChain)

### Hawkins Interdimensional Agent

A LangChain agent from the Downside-Up Complaint Bureau! that handles complaints about inconsistencies in the Normal Objects universe using creative tool chaining, few-shot prompting, and live usage tracking.

---

## File map

```
.
├── normalobjects_langchain.py      # Main file: agent calling tools, prompt, executor, tracker, test runner and analysis.
├── test-lc.py                      # Testing with 2 complaints 
└── Testing agent with complains.md # output from test-lc.py
└── Final output sample.md          # output from the main file
└── lab_summary.md                  # Narrative summary of the lab
```

---

## Key components inside `normalobjects_langchain.py`

| Component | What it does |
|---|---|
| `tools` | Four LangChain tools: `consult_demogorgon`, `check_hawkins_records`, `cast_interdimensional_spell`, `gather_party_wisdom` |
| `few_shot_prompt` | Two example exchanges that model vivid, story-driven investigation tone |
| `prompt` | Full `ChatPromptTemplate`: system message → few-shot examples → chat history → user input → scratchpad |
| `agent_executor` | `AgentExecutor` with `max_iterations=5` and the tracker wired in via `callbacks` |
| `ToolUsageTracker` | `BaseCallbackHandler` subclass that captures per-complaint tool sequences and computes chaining statistics |
| `handle_complaint()` | Runs one complaint, brackets it with `begin_complaint()` / `end_complaint()` for sequence tracking |
| `print_analysis()` | Prints total calls, per-tool counts, most-used tool, chaining rate, per-complaint sequences and total sequence examples |
