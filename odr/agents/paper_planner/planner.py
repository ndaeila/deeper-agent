"""Paper outline planner node logic."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from odr.agents.paper_planner.contracts import PaperOutline


def plan_outline(input_text: str, llm: BaseChatModel) -> PaperOutline:
    """Plan a paper outline (sections/goals/questions) for a user query.

    Args:
        input_text: The user query/task.
        llm: LLM used to produce the outline.

    Returns:
        PaperOutline model.
    """
    
    
    # ===============================
    # SYSTEM PROMPT 
    # ===============================
    system_prompt = """You are a research planner that researches entities in-depth for information gathering purposes akin to a background check.

Given a user query, produce an outline for the research paper that will later be researched via a Retriever agent.

Requirements:
- 4 to 7 sections.
- Each section must have:
  - a clear title
  - a goal
  - 3 to 7 concrete research questions
  - a focused retrieval_query to send to the Retriever for evidence gathering
  - (Optional) requirements: a dictionary of requirements (e.g. {"min_unique_urls": 2}) if deep research is needed.
- Avoid redundant sections.
- The outline should be interesting and "research paper"-like, but not speculative.
- Depending on the user query, consider how you would research the entity or topic you're investigating in depth. 
- You should be able to research the entity or topic in depth, including their other attributes, connections, relationships and affiliations.
- But instead of being broad and general, you should be very specific and detailed about what needs to be researched for that entity.
- Think about what would be commonly found in either an investigation report or a background check report given the entity.
- Do not be afraid to add sections that are not directly related to the entity, but are relevant to understanding specific aspects of the entity (e.g. for a person, like work history, relationships, education, family, etc.).
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=input_text),
    ]

    with_structured = getattr(llm, "with_structured_output", None)
    if callable(with_structured):
        try:
            runner = cast(Any, with_structured(PaperOutline, method="function_calling"))
        except TypeError:
            runner = cast(Any, with_structured(PaperOutline))
        return cast(PaperOutline, cast(Any, runner).invoke(messages, config={"run_name": "plan_outline"}))

    # Fallback: ask for JSON compatible with PaperOutline.
    fallback_prompt = (
        "Return ONLY JSON matching this schema:\n"
        "{title: str, abstract_goal: str, sections: [{section_id: str, title: str, goal: str, "
        "questions: [str], retrieval_query: str, requirements: object}]}\n"
    )
    response = llm.invoke([SystemMessage(content=fallback_prompt), HumanMessage(content=input_text)], config={"run_name": "plan_outline"})
    content = response.content if isinstance(response.content, str) else str(response.content)
    return PaperOutline.model_validate_json(content)


