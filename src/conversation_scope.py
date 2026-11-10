"""Conservative conversation references, independent of any model or API."""
import re
from dataclasses import dataclass, field

ALIASES = {
    'BMO': r'BMO|Bank of Montreal', 'TD': r'TD|Toronto[- ]Dominion(?: Bank)?',
    'RBC': r'RBC|Royal Bank of Canada', 'BNS': r'BNS|Scotiabank|Bank of Nova Scotia',
    'CIBC': r'CIBC|Canadian Imperial Bank of Commerce',
}


@dataclass
class ConversationState:
    banks: list = field(default_factory=list)
    years: list = field(default_factory=list)
    recent_banks: list = field(default_factory=list)
    questions: list = field(default_factory=list)

    def accept(self, scope):
        self.banks = list(scope['banks'])
        self.years = list(scope['years'])
        self.recent_banks = list(dict.fromkeys(self.banks + self.recent_banks))[:5]
        self.questions = (self.questions + [scope['original']])[-4:]


def resolve(question, state):
    question = question.strip()
    def stop(message):
        return {'ready': False, 'message': message, 'original': question}
    if not question:
        return stop('Please enter a research question.')
    matches = [(m.start(), bank) for bank, pattern in ALIASES.items()
               for m in re.finditer(r'\b(?:' + pattern + r')\b', question, re.I)]
    explicit = list(dict.fromkeys(bank for _, bank in sorted(matches)))
    remaining = question
    for pattern in ALIASES.values():
        remaining = re.sub(r'\b(?:' + pattern + r')\b', '', remaining, flags=re.I)
    if re.search(r'\bBank\s+of\s+[A-Za-z]+|\b[A-Z][a-z]+\s+Bank\b', remaining):
        return stop('That bank is outside the supported scope. Choose RBC, TD, BMO, Scotiabank or CIBC.')
    reference = bool(re.search(r'\b(it|its|their|they|them|the bank|that bank|same bank|former|latter)\b', question, re.I))
    comparison = bool(re.search(r'\b(compare|versus|vs\.?|compared|comparison)\b', question, re.I))
    if re.search(r'\b(former|latter)\b', question, re.I):
        return stop('Please name the bank explicitly for this comparison.')
    banks = explicit.copy()
    if reference and (not explicit or comparison):
        candidates = [b for b in (state.banks if not explicit else state.recent_banks) if b not in explicit]
        if len(candidates) != 1:
            return stop('Which bank do you mean by that reference? Please name it explicitly.')
        banks += candidates
    elif not banks:
        banks = state.banks.copy()
    if not banks:
        return stop('Which bank should I research? Choose RBC, TD, BMO, Scotiabank or CIBC.')
    if comparison and len(banks) < 2 and not re.search(r'\b20\d{2}\b.*\b20\d{2}\b', question):
        return stop('Which other bank or fiscal year should I compare?')
    years = list(dict.fromkeys(int(y) for y in re.findall(r'\b(20\d{2})\b', question))) or state.years.copy()
    if not years:
        return stop('Which fiscal year should I use? The saved corpus covers 2024 and 2025.')
    if any(y not in (2024, 2025) for y in years):
        return stop('That fiscal year is outside the saved corpus. Please choose 2024 or 2025.')
    if len(banks)*len(years) > 8:
        return stop('Please narrow the comparison to at most eight bank and year combinations.')
    # Explicit scope takes precedence; prior user questions only provide topic continuity.
    resolved = (f'Current question: {question}\nValidated banks: {", ".join(banks)}. '
                f'Fiscal years: {", ".join(map(str, years))}. '
                'Resolve references using this validated scope. Do not substitute banks or periods.\n'
                'Prior user questions for topic continuity only (not financial evidence):\n' +
                '\n'.join(state.questions[-3:]))
    retrieval_query = question + ' ' + ' '.join(banks) + ' ' + ' '.join(map(str, years))
    return {'ready': True, 'original': question, 'question': resolved, 'retrieval_query': retrieval_query, 'banks': banks,
            'years': years, 'scopes': [{'bank_code': b, 'fiscal_year': y} for b in banks for y in years]}
