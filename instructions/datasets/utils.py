
def input_to_id_default(doc, split, idx):
    if 'id' in doc:
        return str(doc['id'])
    return f'{split}-{idx}'


def input_to_prompt_medqa(doc):
    letter_options = ['A', 'B', 'C', 'D']
    question = doc['sent1']
    # Include rationale if in dataset (this means either that it was pre-computed or is part of fewshot context)
    explanation_str = doc.get('rationale', '')
    choice_str = '\n'.join([f"{letter_options[i]}) {doc['ending' + str(i)]}" for i in range(len(letter_options))])
    if len(explanation_str) == 0:
        text = f'<<Question:>> {question}\n----\n<<Choices:>>\n{choice_str}\n----\n<<Answer:>>'
    else:
        text = f'<<Question:>> {question}\n----\n<<Choices:>>\n{choice_str}\n----\n<<Explanation:>> {explanation_str}\n----\n<<Answer:>>'
    return text


def input_to_target_medqa(doc) -> str:
    letter_options = ['A', 'B', 'C', 'D']
    return letter_options[doc['label']]


def input_to_prompt_pubmedqa(doc):
    choices = ['yes', 'no', 'maybe']
    letters = ['A', 'B', 'C']
    choice_str = '\n'.join([f'{l}) {c}' for l, c in zip(letters, choices)])

    ctx_lines = []
    assert len(doc['LABELS']) == len(doc['CONTEXTS']) 
    for header, ctx in zip(doc['LABELS'], doc['CONTEXTS']):
        ctx_lines.append(f'### {header}\n{ctx}')
    ctxs = '\n' + '\n\n'.join(ctx_lines)
    question = doc['QUESTION']
    # Include rationale if in dataset (this means either that it was pre-computed or is part of fewshot context)
    explanation_str = doc.get('rationale', '')
    if len(explanation_str) == 0:
        text = f'<<Abstract:>> {ctxs}\n----\n<<Question:>> {question}\n----\n<<Choices:>>\n{choice_str}\n----\n<<Answer:>>'
    else:
        text = f'<<Abstract:>> {ctxs}\n----\n<<Question:>> {question}\n----\n<<Choices:>>\n{choice_str}\n----\n<<Explanation:>> {explanation_str}\n----\n<<Answer:>>'

    return text


def input_to_target_pubmedqa(doc):
    choices = ['yes', 'no', 'maybe']
    letters = ['A', 'B', 'C']
    return letters[choices.index(doc['final_decision'])]


def input_to_target_pubmedqa_artificial(doc):
    choices = ['yes', 'no']
    letters = ['A', 'B']
    return letters[choices.index(doc['final_decision'])]


def input_to_prompt_medmcqa(doc):
    letters = ['A', 'B', 'C', 'D']
    question = doc['question']
    # Include rationale if in dataset (this means either that it was pre-computed or is part of fewshot context)
    explanation_str = doc.get('rationale', '')
    choice_str = '\n'.join([f"{l}) {doc['op' + l.lower()]}" for l in letters])
    if len(explanation_str) == 0:    
        text = f'<<Question:>> {question}\n----\n<<Choices:>>\n{choice_str}\n----\n<<Answer:>>'
    else:
        text = f'<<Question:>> {question}\n----\n<<Choices:>>\n{choice_str}\n----\n<<Explanation:>> {explanation_str}\n----\n<<Answer:>>'
    return text

def input_to_target_medmcqa(doc):
    letters = ['A', 'B', 'C', 'D']
    return letters[doc['cop']]
