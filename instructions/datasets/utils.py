
def form_prompt(question, choices, context=None, add_answer_suffix=True, delim='\n\n'):
    pieces = []

    if context is not None:
        pieces.append(f'# CONTEXT\n{context}')

    pieces.append(f'# QUESTION\n{question}')
    pieces.append(f'# CHOICES\n{choices}')

    if add_answer_suffix:
        pieces.append('# ANSWER\n')

    return delim.join(pieces)


def form_target(answer, explanation):
    # COT-Completion
    if explanation is not None and len(explanation) > 0:
        return f'# EXPLANATION\n{explanation}\n\n# ANSWER\n{answer}'
    else:
        return answer

def input_to_id_default(doc, split, idx):
    if 'id' in doc:
        return str(doc['id'])
    return f'{split}-{idx}'


def input_to_prompt_mmlu(doc, is_cot=False):
    letter_options = ['A', 'B', 'C', 'D']
    question = doc['question']
    choice_str = '\n'.join([f"{letter_options[i]}) {choice}" for i, choice in enumerate(doc['choices'])])

    return form_prompt(question, choice_str, context=None, add_answer_suffix=not is_cot)


def input_to_prompt_medqa(doc, is_cot=False):
    letter_options = ['A', 'B', 'C', 'D']
    question = doc['sent1']
    choice_str = '\n'.join([f"{letter_options[i]}) {doc['ending' + str(i)]}" for i in range(len(letter_options))])

    return form_prompt(question, choice_str, context=None, add_answer_suffix=not is_cot)


def input_to_target_mmlu(doc, explanation) -> str:
    letter_options = ['A', 'B', 'C', 'D']
    label = letter_options[doc['answer']]
    return label, form_target(label, explanation)


def input_to_target_medqa(doc, explanation) -> str:
    letter_options = ['A', 'B', 'C', 'D']
    label = letter_options[doc['label']]
    return label, form_target(label, explanation)


def input_to_prompt_pubmedqa(doc, is_cot=False):
    choices = ['yes', 'no', 'maybe']
    letters = ['A', 'B', 'C']
    choice_str = '\n'.join([f'{l}) {c}' for l, c in zip(letters, choices)])

    ctx_lines = []
    assert len(doc['LABELS']) == len(doc['CONTEXTS']) 
    for header, ctx in zip(doc['LABELS'], doc['CONTEXTS']):
        ctx_lines.append(f'## {header}\n{ctx}')
    ctxs = '\n\n'.join(ctx_lines)
    question = doc['QUESTION']

    return form_prompt(question, choice_str, context=ctxs, add_answer_suffix=not is_cot)


def input_to_target_pubmedqa(doc, explanation):
    choices = ['yes', 'no', 'maybe']
    letters = ['A', 'B', 'C']
    label = letters[choices.index(doc['final_decision'])]
    return label, form_target(label, explanation)


def input_to_target_pubmedqa_artificial(doc, explanation):
    choices = ['yes', 'no']
    letters = ['A', 'B']
    label = letters[choices.index(doc['final_decision'])]
    return label, form_target(label, explanation)


def input_to_prompt_medmcqa(doc, is_cot=False):
    letters = ['A', 'B', 'C', 'D']
    question = doc['question']
    choice_str = '\n'.join([f"{l}) {doc['op' + l.lower()]}" for l in letters])
    return form_prompt(question, choice_str, context=None, add_answer_suffix=not is_cot)


def input_to_target_medmcqa(doc, explanation):
    letters = ['A', 'B', 'C', 'D']
    label = letters[doc['cop']]
    return label, form_target(label, explanation)
