# Autonomous Research Agent

Give it a question. It figures out how to research it, searches the web, and hands you a structured report — while you watch it think in real time.

> Built to understand how AI agents actually work, not just how to call one.

---

## Context

This project is a hands-on illustration of concepts I learnt while following [Stanford CME295 — Transformers & Large Language Models](https://cme295.stanford.edu/syllabus/). The course covers the theory behind attention mechanisms, transformer architectures, fine-tuning, and the emerging field of LLM-based agents.

Building this agent was my way of making those ideas concrete: the planning step maps directly to prompt engineering and chain-of-thought reasoning covered in the course, the ReAct loop is the practical expression of tool-use and autonomous decision-making discussed in the later lectures, and the whole thing runs on an LLM that is itself a product of the transformer architecture the course explains from first principles.

If you are following the same course (or just curious), the file to read alongside the syllabus is `agent.py`.

---

## What it does

1. **Planning** — the agent reads the question and breaks it into 3-5 focused sub-questions that together cover the topic.
2. **Searching** — for each sub-question, it searches DuckDuckGo (no API key needed) and collects the top results.
3. **Summarising** — it reads the search snippets and writes a concise factual summary for each sub-question, citing its sources.
4. **Reporting** — it compiles everything into a clean Markdown report with an executive summary and a references section.
5. **Live thinking** — every single step is printed to the screen as it happens.

---

## Stack

| Component | Tool | Why |
|---|---|---|
| LLM | [Groq](https://console.groq.com) — Llama 3.3 70B | Free tier, no credit card. Serves at ~700 tokens/sec — the speed makes the live demo feel genuinely impressive |
| Web search | [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) | No API key, no rate-limit headache, works out of the box |
| Agent logic | Custom ReAct loop (this repo) | Writing it from scratch to understand exactly what happens. LangGraph and smolagents do the same thing but with more layers on top |
| Interface | [Gradio](https://gradio.app) | A working web UI in ~20 lines |

**ReAct** stands for *Reasoning + Acting*. It's the pattern that underlies virtually every production agent: the model thinks, calls a tool, observes the result, then thinks again. That cycle is the whole `agent.py` file.

---

## Project structure

```
autonomous_research_agent/
│
├── agent.py          # The brain: planning, searching, compiling the report
├── tools.py          # The only tool: web search via DuckDuckGo
├── app.py            # The face: Gradio web interface
│
├── requirements.txt  # Python dependencies
├── .env.example      # Template for API key
└── README.md
```

---

## Getting started

### 1. Clone / download

```bash
git clone <your-repo-url>
cd autonomous_research_agent
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv research-agent
source research-agent/bin/activate      # macOS / Linux
# research-agent\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a free Groq API key

Go to [console.groq.com](https://console.groq.com), sign up (no credit card), and create an API key. 

### 5. Set up your `.env` file

```bash
cp .env.example .env
```

Open `.env` and replace `your_groq_api_key_here` with your actual key:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
```

### 6. Launch

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser. Type a question and press Enter.

---

## Example questions to try

- *What are the main causes of coral reef bleaching?*
- *How does nuclear fusion work and why is it hard to achieve?*
- *What are the pros and cons of universal basic income?*
- *What is the current state of quantum computing?*
- *Why did Silicon Valley Bank collapse in 2023?*

---

## How the agent works (the important part)

The file worth reading is `agent.py`. It implements the **ReAct loop** in about 150 lines:

```
User question
     │
     ▼
┌─────────────┐
│    PLAN     │  LLM: "Break this into sub-questions"  →  list of questions
└──────┬──────┘
       │
       ▼  (for each sub-question)
┌─────────────┐
│    ACT      │  Call web_search(sub_question)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   OBSERVE   │  LLM: "Summarise these search results"  →  summary + sources
└──────┬──────┘
       │
       └──► repeat for next sub-question
       │
       ▼
┌─────────────┐
│   REPORT    │  LLM: "Compile a structured report from all findings"
└─────────────┘
```

No external agent framework is used. Every step is explicit and readable. If something breaks, you know exactly where to look.

---

## Customisation

**Change the model:**  
Edit `DEFAULT_MODEL` in `agent.py`. Other free Groq models: `llama-3.1-8b-instant` (faster, smaller), `mixtral-8x7b-32768` (good at structured output).

**More or fewer sub-questions:**  
In `agent.py → plan()`, adjust the system prompt — ask for "2 to 3" instead of "3 to 5".

**More search results per query:**  
In `agent.py → search_and_summarize()`, change `max_results=5` to `max_results=8`.

**Public sharing link:**  
In `app.py`, change `demo.launch(share=False)` to `demo.launch(share=True)`. Gradio will print a temporary public URL you can share with anyone.

---

## Free tier limits (Groq)

As of early 2025, the Groq free tier allows:
- 14,400 requests per day
- 30 requests per minute
- 6,000 tokens per minute for Llama 3.3 70B

One agent run uses around 3-6 LLM calls (1 for planning + 1 per sub-question + 1 for the report).

---

## Troubleshooting

**`GROQ_API_KEY not found`**: Make sure you created a `.env` file (not just edited `.env.example`) and that it's in the same folder as `agent.py`.

**`duckduckgo_search` errors**: DuckDuckGo occasionally rate-limits rapid consecutive searches. Wait 30 seconds and retry, or reduce `max_results`.

**Slow responses**: Switch the model to `llama-3.1-8b-instant` in `agent.py` (much faster, lower quality).

**Port already in use**: Change the port with `demo.launch(server_port=7861)` in `app.py`.
