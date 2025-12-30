"""Orchestrator for Planner -> per-section Retriever loop -> ResearchPaperWriter."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from odr.agents.paper_pipeline.refiner_tools import (
    AddSection,
    NoChange,
    RemoveSection,
    UpdateSection,
)
from odr.agents.paper_planner import PaperOutline, PaperPlanner
from odr.agents.paper_planner.contracts import PaperSection
from odr.agents.research_paper import ResearchPaperWriter
from odr.agents.retriever.contracts import coerce_evidence_items, unique_urls_from_results
from odr.agents.types import WorkerResultExtended
from odr.factory import DefaultLLMFactory


class DeepResearchPaper:
    """End-to-end orchestrator for paper creation.

    Architecture (no nesting of graphs):
    PaperPlanner -> (loop) Retriever -> ResearchPaperWriter (loop)
    """

    def __init__(
        self,
        llm_factory: DefaultLLMFactory | None = None,
        llm: BaseChatModel | None = None,
        retriever: Any = None,
        planner: Any | None = None,
        writer: Any | None = None,
    ):
        """Initialize the orchestrator.

        Args:
            llm_factory: Factory for creating LLMs.
            llm: LLM used by PaperPlanner and ResearchPaperWriter (if factory not used).
            retriever: Retriever-like agent. Retrieves information for the paper.
        """
        if llm_factory:
            self.llm = llm_factory.get_llm(name="refiner-manager", provider="gemini")
        else:
            self.llm = llm
        
        if not self.llm:
            raise ValueError("Either llm_factory or llm must be provided")

        self.retriever = retriever
        if not self.retriever and llm_factory:
             # Lazy import to avoid circular dependency if possible, though strict deps are fine here
             from odr.agents import Retriever
             self.retriever = Retriever(llm_factory=llm_factory)

        if not self.retriever:
            raise ValueError("Retriever is required")

        self.planner = planner or PaperPlanner(llm_factory=llm_factory, llm=self.llm)
        self.writer = writer or ResearchPaperWriter(llm_factory=llm_factory, llm=self.llm)

    def run(self, input_text: str) -> dict[str, Any]:
        """Run the full pipeline and return artifacts plus final markdown."""
        outline = self.planner.plan(input_text)
        section_packets: list[dict[str, Any]] = []

        all_urls: list[str] = []

        # Use a list that we can modify (Refinement Step)
        pending_sections = list(outline.sections)
        completed_sections_info: list[dict[str, Any]] = []

        while pending_sections:
            section = pending_sections.pop(0)
            
            # 1. Run Retriever for this section
            section_query = _build_section_query(input_text=input_text, outline=outline, section=section)
            retriever_state = self.retriever.run(section_query)

            # 2. Collect results
            worker_results = retriever_state.get("worker_results") or []
            worker_results = [r for r in worker_results if isinstance(r, dict)]
            urls = sorted(unique_urls_from_results(worker_results))
            all_urls.extend(urls)
            evidence_items = _collect_evidence_items(worker_results, limit=100000)

            # 3. Store packet
            compiled_report = retriever_state.get("compiled_report") or ""
            section_packets.append(
                {
                    "section": section.model_dump(mode="json"),
                    "retriever_final_status": retriever_state.get("final_status"),
                    "retriever_compiled_output": retriever_state.get("compiled_output"),
                    "retriever_compiled_report": compiled_report,
                    "unique_urls": urls,
                    "evidence_items": evidence_items,
                }
            )
            completed_sections_info.append(
                {
                    "title": section.title,
                    "goal": section.goal,
                    "report_summary": compiled_report
                }
            )

            # 4. Refine the remaining plan?
            if pending_sections:
                pending_sections = self._refine_outline(
                    input_text=input_text,
                    current_section=section,
                    current_report=compiled_report,
                    pending_sections=pending_sections,
                    completed_sections_info=completed_sections_info,
                )
                # Update the main outline object to reflect changes (for the writer)
                outline.sections = [
                    PaperSection(**p["section"]) for p in section_packets
                ] + pending_sections

        references = _stable_references(sorted(set(all_urls)))
        outline_dict = outline.model_dump(mode="json")
        outline_dict["references"] = references

        paper_state = self.writer.run(
            input_text=input_text,
            outline=outline_dict,
            section_packets=section_packets,
        )
        return {
            "outline": outline_dict,
            "section_packets": section_packets,
            "paper_markdown": paper_state.get("paper_markdown", ""),
            "paper_status": paper_state.get("final_status"),
        }

    def _refine_outline(
        self,
        input_text: str,
        current_section: PaperSection,
        current_report: str,
        pending_sections: list[PaperSection],
        completed_sections_info: list[dict[str, Any]],
    ) -> list[PaperSection]:
        """Reflect on the findings and adjust the remaining sections if needed."""
        print(f"\n[Refiner] Reflecting on '{current_section.title}'...")
        
        tools = [AddSection, RemoveSection, UpdateSection, NoChange]
        refiner = self.llm.bind_tools(tools)
        
        # Prepare context
        completed_txt = "\n".join(
            f"- {s['title']}: {s['goal']}" for s in completed_sections_info
        )
        pending_txt = "\n".join(
            f"- [ID: {s.section_id}] {s.title}: {s.goal}" for s in pending_sections
        )
        
        system_msg = (
            "You are a Research Director overseeing a deep research process.\n"
            "We have just completed research on one section. Based on the findings, "
            "determine if we need to adjust the *remaining* sections of the plan.\n\n"
            "Guidelines:\n"
            "- Add a section if a new, crucial subtopic was discovered.\n"
            "- Remove a section if it's now redundant or covered by the previous research.\n"
            "- Update a section if the focus needs to shift based on new information.\n"
            "- Call NoChange if the current plan is still solid.\n"
            "- Be conservative: only change if necessary."
        )
        
        user_msg = (
            f"Original Query: {input_text}\n\n"
            f"Already Researched:\n{completed_txt}\n\n"
            f"Just Finished Section: {current_section.title}\n"
            f"Findings Summary:\n{current_report}\n\n"
            f"Remaining Plan:\n{pending_txt}\n\n"
            "Do we need to modify the Remaining Plan? Use tools to make changes or confirm NoChange."
        )
        
        try:
            response = refiner.invoke([SystemMessage(content=system_msg), HumanMessage(content=user_msg)], config={"run_name": "refine_outline"})
            tool_calls = response.tool_calls
        except Exception as e:
            print(f"[Refiner] Error during refinement: {e}")
            return pending_sections

        if not tool_calls:
            print("[Refiner] No tool calls, continuing...")
            return pending_sections

        # Apply changes
        new_pending = list(pending_sections)
        
        for call in tool_calls:
            name = call["name"]
            args = call["args"]
            
            if name == "NoChange":
                print(f"[Refiner] No changes: {args.get('reason')}")
                continue
                
            if name == "AddSection":
                print(f"[Refiner] Adding section: {args['title']}")
                import uuid
                new_sec = PaperSection(
                    section_id=f"gen_{uuid.uuid4().hex[:8]}",
                    title=args["title"],
                    goal=args["goal"],
                    questions=args["questions"],
                    retrieval_query=args["retrieval_query"],
                )
                
                # Find insertion point
                inserted = False
                if args.get("after_section_id"):
                    for i, s in enumerate(new_pending):
                        if s.section_id == args["after_section_id"]:
                            new_pending.insert(i + 1, new_sec)
                            inserted = True
                            break
                if not inserted:
                    new_pending.append(new_sec)

            elif name == "RemoveSection":
                print(f"[Refiner] Removing section: {args['section_id']}")
                new_pending = [s for s in new_pending if s.section_id != args["section_id"]]

            elif name == "UpdateSection":
                print(f"[Refiner] Updating section: {args['section_id']}")
                for s in new_pending:
                    if s.section_id == args["section_id"]:
                        if args.get("title"): s.title = args["title"]
                        if args.get("goal"): s.goal = args["goal"]
                        if args.get("questions"): s.questions = args["questions"]
                        if args.get("retrieval_query"): s.retrieval_query = args["retrieval_query"]

        return new_pending


def _build_section_query(input_text: str, outline: PaperOutline, section: Any) -> str:
    questions = "\n".join(f"- {q}" for q in (section.questions or []))
    return (
        f"User query:\n{input_text}\n\n"
        f"Paper title:\n{outline.title}\n\n"
        f"Section to research:\n{section.title}\n\n"
        f"Section goal:\n{section.goal}\n\n"
        f"Research questions:\n{questions}\n\n"
        f"Focused retrieval query:\n{section.retrieval_query}\n"
    )


def _stable_references(urls: list[str]) -> list[dict[str, Any]]:
    return [{"n": idx + 1, "url": u} for idx, u in enumerate(urls)]


def _collect_evidence_items(results: list[WorkerResultExtended], limit: int = 100000) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in results:
        for ev in coerce_evidence_items(r.get("evidence")):
            url = ev.get("url") or ev.get("source_url")
            if not url:
                continue
            out.append(
                {
                    "url": url,
                    "title": ev.get("title"),
                    "excerpt": ev.get("excerpt"),
                }
            )
            if len(out) >= limit:
                return out
    return out


