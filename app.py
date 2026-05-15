"""
Run with:
    python app.py

Then open http://localhost:7860 in your browser.

The UI has two panels:
  - Left : "Thinking" — every step the agent takes, in real time
  - Right: "Report"   — the final structured Markdown report

Live thinking panel to watch the agent plan, search, and reason.
"""
import gradio as gr
from agent import ResearchAgent


# We create one shared agent instance. Groq is stateless so this is safe.
agent = ResearchAgent()


def run_research(topic: str):
    """
    Generator function that drives the Gradio UI update loop.

    Gradio supports generator functions out of the box: every time we
    `yield`, it pushes an update to the browser without a page reload.
    We yield a tuple (thinking_log, report) after each step so both
    panels update progressively.

    Args:
        topic: The research question typed by the user.

    Yields:
        (thinking_log: str, report: str)
    """
    if not topic.strip():
        yield "Please enter a research topic.", ""
        return

    thinking_log = []
    final_report  = ""

    def on_step(step_text: str):
        """
        Callback called by the agent after each action.
        We append the new step to the log and yield immediately
        so the UI updates without waiting for the full run to finish.
        """
        thinking_log.append(step_text)

    # We can't yield from inside on_step (different call frame), so we run
    # the agent in a slightly different way: we call plan() manually and
    # hook into the generator pattern by running step by step.
    #
    # Simpler approach that works with Gradio: run the whole agent but
    # use a mutable list as a message queue and poll it.
    # Even simpler: just yield after each on_step by restructuring the loop.

    # Step 1: planning
    yield "Breaking your question into sub-questions...", ""

    sub_questions = agent.plan(topic)
    thinking_log.append(
        "**Research plan:**\n" + "\n".join(f"- {q}" for q in sub_questions)
    )
    yield "\n\n---\n\n".join(thinking_log), ""

    # Step 2: search and summarise, one sub-question at a time
    findings = []
    for i, question in enumerate(sub_questions, start=1):
        thinking_log.append(f"**[{i}/{len(sub_questions)}] Searching:** _{question}_")
        yield "\n\n---\n\n".join(thinking_log), ""

        finding = agent.search_and_summarize(question)
        findings.append(finding)

        source_lines = "\n".join(
            f"  - [{s['title']}]({s['url']})"
            for s in finding["sources"][:3]
        )
        thinking_log.append(
            f"**[{i}/{len(sub_questions)}] Summary:**\n{finding['summary']}"
            + (f"\n\nSources:\n{source_lines}" if source_lines else "")
        )
        yield "\n\n---\n\n".join(thinking_log), ""

    # --- Step 3: compile the final report ---
    thinking_log.append("**Compiling final report...**")
    yield "\n\n---\n\n".join(thinking_log), ""

    final_report = agent.compile_report(topic, findings)

    thinking_log.append("**Done. Report is ready.**")
    yield "\n\n---\n\n".join(thinking_log), final_report


# Builds the UI
with gr.Blocks(
    title="Autonomous Research Agent",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # 🔍 Autonomous Research Agent
        Type a topic or question below. The agent will plan its own research,
        search the web, and write a structured report — showing its reasoning at every step.

        **Powered by:** Groq (Llama 3.3 70B) · DuckDuckGo Search · LangGraph-free ReAct loop
        """
    )

    with gr.Row():
        topic_input = gr.Textbox(
            label="Research topic or question",
            placeholder='e.g. "What are the environmental impacts of lithium mining?"',
            lines=2,
            scale=4,
        )
        run_button = gr.Button("Research →", variant="primary", scale=1)

    gr.Markdown("---")

    with gr.Row():
        thinking_output = gr.Markdown(
            label="Agent thinking (live)",
            value="*The agent's reasoning will appear here step by step...*",
            elem_id="thinking-panel",
        )
        report_output = gr.Markdown(
            label="Final report",
            value="*The structured report will appear here when research is complete.*",
            elem_id="report-panel",
        )

    # button connected to generator function
    run_button.click(
        fn=run_research,
        inputs=[topic_input],
        outputs=[thinking_output, report_output],
    )

    # allows pressing Enter in the text box
    topic_input.submit(
        fn=run_research,
        inputs=[topic_input],
        outputs=[thinking_output, report_output],
    )

    gr.Markdown(
        """
        ---
        *Sources are cited in the report. The agent searches the live web, so results
        may vary. Always verify important claims.*
        """
    )


if __name__ == "__main__":
    # share=False keeps it local; set to True to get a public Gradio link
    demo.launch(share=False)
