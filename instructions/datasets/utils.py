
def form_prompt(question, choices, context=None, explanation=None, delim='\n\n'):
    pieces = []

    if context is not None:
        pieces.append(f'# CONTEXT\n{context}')

    pieces.append(f'# QUESTION\n{question}')
    pieces.append(f'# CHOICES\n{choices}')

    if explanation is not None and len(explanation) > 0:
        pieces.append(f'# EXPLANATION\n{explanation}')

    pieces.append('# ANSWER\n')

    return delim.join(pieces)
    

def input_to_id_default(doc, split, idx):
    if 'id' in doc:
        return str(doc['id'])
    return f'{split}-{idx}'


def input_to_prompt_medqa(doc, explanation):
    letter_options = ['A', 'B', 'C', 'D']
    question = doc['sent1']
    choice_str = '\n'.join([f"{letter_options[i]}) {doc['ending' + str(i)]}" for i in range(len(letter_options))])

    return form_prompt(question, choice_str, context=None, explanation=explanation)


def input_to_target_medqa(doc) -> str:
    letter_options = ['A', 'B', 'C', 'D']
    return letter_options[doc['label']]


def input_to_prompt_pubmedqa(doc, explanation):
    choices = ['yes', 'no', 'maybe']
    letters = ['A', 'B', 'C']
    choice_str = '\n'.join([f'{l}) {c}' for l, c in zip(letters, choices)])

    ctx_lines = []
    assert len(doc['LABELS']) == len(doc['CONTEXTS']) 
    for header, ctx in zip(doc['LABELS'], doc['CONTEXTS']):
        ctx_lines.append(f'## {header}\n{ctx}')
    ctxs = '\n\n'.join(ctx_lines)
    question = doc['QUESTION']

    return form_prompt(question, choice_str, context=ctxs, explanation=explanation)


def input_to_target_pubmedqa(doc):
    choices = ['yes', 'no', 'maybe']
    letters = ['A', 'B', 'C']
    return letters[choices.index(doc['final_decision'])]


def input_to_target_pubmedqa_artificial(doc):
    choices = ['yes', 'no']
    letters = ['A', 'B']
    return letters[choices.index(doc['final_decision'])]


def input_to_prompt_medmcqa(doc, explanation):
    letters = ['A', 'B', 'C', 'D']
    question = doc['question']
    choice_str = '\n'.join([f"{l}) {doc['op' + l.lower()]}" for l in letters])
    return form_prompt(question, choice_str, context=None, explanation=explanation)


def input_to_target_medmcqa(doc):
    letters = ['A', 'B', 'C', 'D']
    return letters[doc['cop']]
