# Hallucination case analysis

Faithfulness ≤2 flags a material grounding risk under the LLM rubric. These cases require human confirmation.

## agentic eval_26

Question: What were Bank of Atlantis credit losses in 2025?

Faithfulness: 1/5. The answer refers to credit losses for Scotiabank instead of the requested Bank of Atlantis, which is outside the corpus. Therefore, the claims are not supported by the correct bank context.

Unsupported claims: 

Evidence inspected: BNS_2025_annual_report_FY_00066, BNS_2025_annual_report_FY_00389, BNS_2025_annual_report_FY_00060, BNS_2025_annual_report_FY_00390, BNS_2025_AIF_FY_00044, BNS_2025_AIF_FY_00041, BNS_2025_AIF_FY_00022, BNS_2025_AIF_FY_00042, BNS_2025_quarterly_Q1_00129, BNS_2025_quarterly_Q1_00134, BNS_2025_quarterly_Q1_00130, BNS_2025_quarterly_Q1_00133, BNS_2025_quarterly_Q2_00146, BNS_2025_quarterly_Q2_00062, BNS_2025_quarterly_Q2_00152, BNS_2025_quarterly_Q2_00045, BNS_2025_quarterly_Q3_00150, BNS_2025_quarterly_Q3_00054, BNS_2025_quarterly_Q3_00151, BNS_2025_quarterly_Q3_00035

Observed routing failure: the question names unsupported Bank of Atlantis, but the agent answered with Scotiabank figures. An unrestricted tool search substituted another bank. Add an explicit unknown-bank gate and verify the final answer entity against the requested bank before returning it.The judge explanation identifies the specific claim; confirm it against the saved context before assigning a definitive cause.

Remediation: require supporting citations for each disputed claim, retrieve missing banks/periods, and use deterministic arithmetic with cited inputs; abstain when either input is absent.

## basic_top_k eval_08

Question: What interest-rate risk management did BMO disclose in its 2025 annual report?

Faithfulness: 1/5. The answer claims insufficient information to address the question, which is accurate given the lack of specific disclosures on interest-rate risk management in the provided context.

Unsupported claims: 

Evidence inspected: BMO_2025_annual_report_FY_00124, BMO_2025_annual_report_FY_00021, BMO_2025_annual_report_FY_00143, BMO_2025_annual_report_FY_00189, BMO_2025_annual_report_FY_00237

Judge inconsistency: this is the fixed insufficient-information refusal, with no affirmative financial claim. The rubric assigns a faithful refusal 5 even when completeness is low. Treat this flag as a likely grader false positive; retain the raw score and request independent adjudication.The judge explanation identifies the specific claim; confirm it against the saved context before assigning a definitive cause.

Remediation: require supporting citations for each disputed claim, retrieve missing banks/periods, and use deterministic arithmetic with cited inputs; abstain when either input is absent.

## mmr_structured eval_28

Question: Did the bank improve its risk profile?

Faithfulness: 2/5. The claim that the Toronto-Dominion Bank improved its risk profile is not fully supported by the evidence provided. While the update to the Enterprise Risk Framework is mentioned, the context does not explicitly state that this constitutes an improvement in the risk profile, leading to a material unsupported claim.

Unsupported claims: The bank improved its risk profile without sufficient evidence to support this claim.; The assertion of a proactive approach to risk management is not substantiated by the evidence provided.

Evidence inspected: TD_2025_quarterly_Q3_00095, BNS_2024_quarterly_Q3_00074, TD_2025_annual_report_FY_00314, TD_2024_annual_report_FY_00325, BNS_2024_quarterly_Q2_00006

Ambiguity handling needs review: the request specifies neither bank nor period. Retrieval and synthesis can select an arbitrary institution instead of asking for clarification. Add a clarification route before retrieval and assess the saved answer against this expected behavior.The judge explanation identifies the specific claim; confirm it against the saved context before assigning a definitive cause.

Remediation: require supporting citations for each disputed claim, retrieve missing banks/periods, and use deterministic arithmetic with cited inputs; abstain when either input is absent.
