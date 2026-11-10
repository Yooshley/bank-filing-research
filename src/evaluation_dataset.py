"""Thirty fixed questions, coverage scopes and answer expectations; no invented gold facts."""


def make_dataset(latest=2025, previous=2024):
    questions = []
    topics = [
        ('RBC', 'credit', 'credit risks and provisions for credit losses'),
        ('RBC', 'market', 'market risk and Value-at-Risk measures'),
        ('RBC', 'climate', 'physical and transition climate risks'),
        ('TD', 'operational', 'cybersecurity and operational risks'),
        ('TD', 'credit', 'residential mortgage credit risks'),
        ('TD', 'liquidity', 'liquidity and funding risks'),
        ('BMO', 'liquidity', 'liquidity risk controls and liquidity coverage measures'),
        ('BMO', 'market', 'interest-rate risk management'),
        ('BMO', 'operational', 'operational and technology risks'),
        ('BNS', 'climate', 'physical and transition climate risks'),
        ('BNS', 'credit', 'credit risk and loan-loss provisions'),
        ('BNS', 'market', 'market risk management'),
        ('CIBC', 'credit', 'housing and mortgage credit risks'),
        ('CIBC', 'operational', 'cybersecurity and technology risks'),
        ('CIBC', 'liquidity', 'liquidity risk management'),
    ]
    names = {'RBC': 'RBC', 'TD': 'TD', 'BMO': 'BMO', 'BNS': 'Scotiabank', 'CIBC': 'CIBC'}
    def add(question, kind, banks, risk, scopes, expected, reference_query=None):
        questions.append({'id': f'eval_{len(questions)+1:02d}', 'question': question,
                          'question_type': kind, 'banks': banks, 'risk': risk,
                          'scopes': scopes, 'expected_behavior': expected,
                          'reference_query': reference_query or question})
    def scope(bank, year):
        return {'bank_code': bank, 'fiscal_year': year, 'document_type': 'annual_report'}
    for bank, risk, topic in topics:
        add(f'What {topic} did {names[bank]} disclose in its {latest} annual report?',
            'factual', [bank], risk, [scope(bank, latest)],
            f'Identify the disclosed {topic}; cite the correct bank/year. Do not invent amounts absent from evidence.')
    comparisons = [
        ('RBC', 'TD', 'credit', 'credit risk management'),
        ('TD', 'BMO', 'credit', 'residential mortgage credit risk'),
        ('BMO', 'CIBC', 'liquidity', 'liquidity risk controls'),
        ('BNS', 'RBC', 'climate', 'physical and transition climate risks'),
    ]
    for left, right, risk, topic in comparisons:
        add(f'Compare {names[left]} and {names[right]} disclosures on {topic} in their {latest} annual reports.',
            'comparative', [left, right], risk, [scope(left, latest), scope(right, latest)],
            'Describe supported similarities and differences with evidence from both banks; state missing evidence.')
    add(f'Compare how all five Canadian banks describe AI model risk or OSFI E-23 in their {latest} annual reports.',
        'comparative', list(names), 'model_risk', [scope(bank, latest) for bank in names],
        'Cover all five banks, cite supporting excerpts, and distinguish no retrieved mention from no disclosure.',
        'artificial intelligence model risk OSFI E-23 governance validation')
    for bank, topic, risk in [('RBC', 'provision for credit losses', 'credit'),
                              ('TD', 'cybersecurity risk', 'operational'),
                              ('BMO', 'liquidity risk', 'liquidity'),
                              ('BNS', 'climate risk', 'climate'),
                              ('CIBC', 'mortgage credit risk', 'credit')]:
        add(f'How did {names[bank]} disclosures about {topic} change between {previous} and {latest}?',
            'trend', [bank], risk, [scope(bank, previous), scope(bank, latest)],
            'Compare evidence from both years. Do not infer a new risk merely from its absence in a retrieved sample.')
    add(f'What were Bank of Atlantis credit losses in {latest}?', 'edge', [], 'credit',
        [{'bank_code': 'ATLANTIS'}], 'State that this bank is outside the corpus; do not fabricate figures.')
    add('What will RBC provision for credit losses be in 2030?', 'edge', ['RBC'], 'credit',
        [{'bank_code': 'RBC', 'fiscal_year': 2030}], 'Decline to predict an unsupported future filing or number.')
    add('Did the bank improve its risk profile?', 'edge', [], 'ambiguous', [],
        'Ask which bank and period; do not pick an arbitrary bank or claim improvement.')
    add(f'Quels risques de crédit la RBC décrit-elle dans son rapport annuel {latest} ?', 'edge', ['RBC'], 'credit',
        [scope('RBC', latest)], 'Answer in French using supported RBC credit-risk disclosures and citations.',
        'RBC credit risk management provision credit losses')
    add(f'Calculate the percentage change in RBC provision for credit losses from {previous} to {latest}.',
        'edge', ['RBC'], 'calculation', [scope('RBC', previous), scope('RBC', latest)],
        'Use matching annual PCL amounts with units and citations; show (new-old)/old*100, or abstain if either input is missing.',
        'RBC total provision for credit losses fiscal year')
    assert len(questions) == 30
    return questions
