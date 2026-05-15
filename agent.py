"""
How it works (the ReAct pattern):
  1. PLAN   — the LLM reads your question and breaks it into sub-questions.
  2. ACT    — for each sub-question, we call the search tool.
  3. OBSERVE— we feed the raw search results back to the LLM to summarise.
  4. REPEAT — steps 2-3 run for every sub-question.
  5. REPORT — once all sub-questions are answered, the LLM writes a
              structured final report with cited sources.

This is exactly what LangGraph, smolagents and other "agent framework"
do under the hood. Writing it makes it much easier to debug and
understand what is actually happening.
"""

import json
import os

from groq import Groq
from dotenv import load_dotenv

from tools import web_search, format_search_results

load_dotenv()


def _chat(client: Groq, messages: list[dict], model: str, temperature: float = 0.3) -> str:
    """
    Send a list of messages to Groq and return the text of the first choice.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


class ResearchAgent:
    """
    Autonomous research agent that plans, searches, and synthesises findings.

    Usage:
        agent = ResearchAgent()
        report = agent.run("What are the main causes of coral reef bleaching?")
        print(report)

    For a live UI, pass a callback that the agent calls after every step:
        def my_callback(step_text: str):
            print(step_text)

        agent.run("...", on_step=my_callback)
    """

    # Llama 3.3 70B is the most capable free model on Groq.
    # If you hit rate limits, swap to "llama-3.1-8b-instant" — much faster.
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, model: str | None = None):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Copy .env.example to .env and add your key."
            )
        self.client = Groq(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL

    # Step 1 — Plan
    def plan(self, topic: str) -> list[str]:
        """
        Ask the LLM to decompose a broad topic into 3-5 focused sub-questions.

        We ask for JSON output so it's easy to parse. The system prompt is
        strict about format to avoid having to write a fragile parser.

        Args:
            topic: The research question given by the user.

        Returns:
            A list of sub-question strings.
        """
        system = (
            "You are a research assistant that helps plan structured research. "
            "Your job is to break a broad research topic into a set of focused, "
            "independent sub-questions that together cover the topic well.\n\n"
            "Rules:\n"
            "- Return ONLY a JSON array of strings, nothing else.\n"
            "- 3 to 5 sub-questions, each short and searchable.\n"
            "- Each sub-question should target a different angle of the topic.\n"
            "- No numbering, no bullet points — just a JSON array.\n\n"
            'Example output: ["What is X?", "How does X work?", "What are the risks of X?"]'
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Topic: {topic}"},
        ]

        raw = _chat(self.client, messages, self.model)

        # Parse the JSON array. If the model sneaks in extra text, strip it.
        try:
            # Find the first '[' and last ']' in case there's surrounding text
            start = raw.index("[")
            end   = raw.rindex("]") + 1
            sub_questions = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            # Fallback: treat every non-empty line as a sub-question
            sub_questions = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]

        return sub_questions[:5]  # cap at 5 to stay within rate limits

    # Step 2 — Search and summarise (one sub-question at a time)
    def search_and_summarize(self, sub_question: str) -> dict:
        """
        Search the web for one sub-question, then ask the LLM to summarise
        what it found in a few sentences with citations.

        Args:
            sub_question: A focused question, e.g. "What causes coral bleaching?"

        Returns:
            A dict with keys:
              - sub_question : the original question
              - summary      : LLM-generated summary (2-4 sentences)
              - sources      : list of {title, url} dicts used as citations
        """
        results = web_search(sub_question, max_results=5)
        formatted = format_search_results(results)

        system = (
            "You are a careful research assistant. "
            "You are given a question and a set of web search results. "
            "Write a concise, factual answer (3-5 sentences) to the question "
            "based only on the provided search results. "
            "Do not invent facts. If the results don't answer the question, say so. "
            "At the end, list the source numbers you relied on in brackets, "
            "like: (Sources: [1], [3])."
        )

        user_msg = (
            f"Question: {sub_question}\n\n"
            f"Search results:\n{formatted}"
        )

        summary = _chat(self.client, [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ], self.model)

        # Keep the sources that actually had content
        sources = [
            {"title": r["title"], "url": r["url"]}
            for r in results if r["url"]
        ]

        return {
            "sub_question": sub_question,
            "summary":      summary,
            "sources":      sources,
        }

    # Step 3 — Compile the final report
    def compile_report(self, topic: str, findings: list[dict]) -> str:
        """
        Take all the per-sub-question findings and produce one structured report.

        The report has:
          - An executive summary
          - One section per sub-question
          - A references list at the bottom

        Args:
            topic:    The original user question.
            findings: List of dicts from search_and_summarize().

        Returns:
            A Markdown-formatted report string.
        """
        # Build a condensed brief for the LLM so it doesn't need to re-read
        # all the raw search results (saves tokens)
        brief_parts = []
        for f in findings:
            brief_parts.append(f"### {f['sub_question']}\n{f['summary']}")
        brief = "\n\n".join(brief_parts)

        # Collect all unique sources for the reference list
        seen_urls = set()
        all_sources = []
        ref_num = 1
        for f in findings:
            for s in f["sources"]:
                if s["url"] not in seen_urls:
                    seen_urls.add(s["url"])
                    all_sources.append(f"[{ref_num}] {s['title']} — {s['url']}")
                    ref_num += 1

        sources_block = "\n".join(all_sources)

        system = (
            "You are a professional research analyst. "
            "You receive research notes covering different aspects of a topic. "
            "Write a clear, structured Markdown report. "
            "Use this exact structure:\n"
            "1. A short title (H1)\n"
            "2. An 'Executive Summary' section (3-5 sentences)\n"
            "3. One section per research finding (H2 heading = the sub-question)\n"
            "4. A 'References' section at the end with the provided source list\n\n"
            "Keep the tone informative but accessible. "
            "Do not add information beyond what is in the notes."
        )

        user_msg = (
            f"Original question: {topic}\n\n"
            f"Research notes:\n{brief}\n\n"
            f"Sources to include in the References section:\n{sources_block}"
        )

        report = _chat(self.client, [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ], self.model, temperature=0.4)

        return report


    def run(self, topic: str, on_step=None) -> str:
        """
        Full agent pipeline: plan → search → summarise → report.

        The optional `on_step` callback is called after each meaningful
        action with a human-readable description of what just happened.
        This is what makes the "thinking" visible in the UI.

        Args:
            topic:   The research question to investigate.
            on_step: Optional callable(step_text: str). Called after each step.

        Returns:
            The final Markdown report as a string.
        """
        def emit(text: str):
            """Helper so we don't have to check for None every time."""
            if on_step:
                on_step(text)

        # ---- Planning ----
        emit(f"**Planning research for:** {topic}\n\nBreaking the topic into sub-questions...")
        sub_questions = self.plan(topic)

        emit(
            "**Research plan ready.** I will investigate:\n"
            + "\n".join(f"  - {q}" for q in sub_questions)
        )

        # Searching and summarising
        findings = []
        for i, question in enumerate(sub_questions, start=1):
            emit(f"**[{i}/{len(sub_questions)}] Searching:** {question}")
            finding = self.search_and_summarize(question)
            findings.append(finding)

            source_list = "\n".join(
                f"    - [{s['title']}]({s['url']})" for s in finding["sources"][:3]
            )
            emit(
                f"**[{i}/{len(sub_questions)}] Done.** Summary:\n"
                f"{finding['summary']}\n\n"
                f"Sources consulted:\n{source_list}"
            )

        # Compiling the final report
        emit("**All sub-questions answered.** Compiling the final report...")
        report = self.compile_report(topic, findings)

        emit("**Research complete.**")
        return report
