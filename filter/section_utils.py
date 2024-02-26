
import regex as re

# MIMIC

def default_section_splitter(text):
    return re.split(r'\n{2,}', text)


def sections_to_str(sections):
    return '\n\n'.join(sections)
