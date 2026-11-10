# Executive summary

Recommendation: **iterate further under analyst supervision**. The system searches public Canadian bank filings and drafts cited answers. It is a research aid; final compliance conclusions remain with an analyst.

The evaluation is complete: 90 of 90 planned answers have been judged, using the same 30 questions across basic retrieval, diverse retrieval with structured output, and a tool-using agent. The strongest complete configuration by faithfulness, relevance and completeness is mmr_structured: 4.57/5 faithfulness, 4.67/5 relevance, and 4.43/5 completeness. Its estimated operating API cost is US$0.000768 per query, or US$280.49 per year at 1,000 queries every day.

The automated judge flagged 3 answers with faithfulness at or below 2/5. These are unconfirmed flags; 3 fixed-refusal scores also conflict with the stated grading rubric. Review these cases before any deployment. Human accuracy and citation reviews from steps 4 and 5 are still unfilled, and automated scores do not replace them. The judge uses the same model family as the answer generator, so correlated errors are possible.

The corpus contains 44 distinct filings and 10,196 indexed chunks. Some quarterly coverage is missing; a claim about a new risk cannot be established when an earlier filing is absent. Search excerpts cannot prove a topic is absent from an entire report.

The cost estimate covers measured API tokens. It excludes hosting, local computation, analyst review and setup. No timed analyst study has been performed, so the proposed 50% reduction in review time has not been demonstrated. Complete human checks, repair low-faithfulness cases, fill data gaps and measure analyst task time before a supervised pilot.

Detailed results: [evaluation report](evaluation_report.md), [system documentation](system_documentation.md), and [master comparison](evaluation/54a68bec798eab0e/master_comparison.csv).
