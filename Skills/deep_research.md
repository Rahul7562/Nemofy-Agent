---
name: deep_research
description: Multi-source research methodology covering evidence evaluation, fact verification, source attribution, synthesis, and structured report generation.
---

# Deep Research Skill

Use this skill for open-ended investigations, technical comparisons, market or literature reviews, competitive audits, and fact-sensitive reports.

## 1. Multi-Source Research Strategy
- Decompose the research question into 3-6 specific sub-questions before fetching sources.
- Consult multiple independent primary sources (official documentation, specifications, papers, primary data) rather than relying on a single secondary summary.
- When gathering web pages with `web_fetch`, extract structured text and record the exact URL and access context.
- For large investigations (>4 independent topics), delegate bounded sub-questions to isolated worker sub-agents via `spawn_subagent`.

## 2. Evidence Evaluation
- Rank sources by reliability:
  1. Primary specifications, peer-reviewed papers, official engineering docs, raw datasets.
  2. Reputable technical analyses and benchmark reports with reproducible methodology.
  3. Blog posts, commentary, and forums (corroborate before citing as fact).
- Explicitly separate **verified facts**, **reasonable inferences**, and **open uncertainties**.
- Flag conflicts across sources instead of silently picking one side.

## 3. Fact Verification
- Cross-check quantitative claims (dates, versions, pricing, limits, benchmark numbers) across at least two independent sources whenever available.
- Verify mathematical computations, percentages, and unit conversions using Python in the sandbox rather than mental arithmetic.
- Never fabricate citations, URLs, quotes, or statistics. If a claim cannot be verified, explicitly mark it as unverified.

## 4. Source Attribution
- Attach inline source references (URL or workspace file path) to every non-trivial factual claim.
- Include a consolidated **Sources** section at the end of the report listing title, URL/path, and what claim it supports.

## 5. Research Synthesis & Report Generation
- Organize deliverables into:
  1. **Executive Summary** (3-5 crisp takeaways answering the core question directly).
  2. **Methodology & Scope** (what was examined and any constraints or gaps).
  3. **Key Findings** (structured by theme with supporting evidence and trade-off tables).
  4. **Conflicts, Risks & Limitations** (contradictory evidence and unknowns).
  5. **Recommendations / Next Steps** (actionable, prioritized).
  6. **Sources** (verified attributions).
- Save long reports to `/workspace/` using `write_file` when requested, and summarize concisely in the final response.
