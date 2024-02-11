import re


def clean_mimic(text, has_identifiers=True):
    """
    :param text: string representing raw MIMIC note
    :return: cleaned string
    - Replace [**Patterns**] with spaces
    """
    text = text.strip()
    text = re.sub(r'\n{2,}', '\nDOUBLENEWLINE\n', text)
    cleaned = []
    for line in text.split('\n'):
        line_strip = line.strip()
        if has_identifiers:
            line_strip = remove_identifiers(line_strip)
        line_strip = re.sub(r'\s+', ' ', line_strip).strip()
        if len(line_strip) > 0:
            cleaned.append(line_strip)
    cleaned_str = '\n'.join(cleaned)
    cleaned_str = cleaned_str.replace('DOUBLENEWLINE', '')
    # BRIEF HOSPITAL COURSE     : patient ... -> BRIEF HOSPITAL COURSE: patient ...
    cleaned_str_no_space_before_colon = re.sub(r'\s+:', ':', cleaned_str)
    return cleaned_str_no_space_before_colon.strip()


def remove_identifiers(text, replacement='___'):
    '''
    Removes MIMIC markers of deidentified info from the text.
    Replaces with single, all-caps words such as ID, AGE, PHONE, etc.
    '''
    regex = r"\[\*\*.{0,15}%s.*?\*\*\]"

    text = re.sub(regex % "serial number", replacement, text, flags=re.I)
    text = re.sub(regex % "identifier", replacement, text, flags=re.I)
    text = re.sub(regex % "medical record number", replacement, text, flags=re.I)
    text = re.sub(regex % "social security number", replacement, text, flags=re.I)

    # AGE
    text = re.sub(regex % "age", replacement, text, flags=re.I)

    # PHONE
    text = re.sub(regex % "phone", replacement, text, flags=re.I)
    text = re.sub(regex % "pager number", replacement, text, flags=re.I)
    text = re.sub(regex % "contact info", replacement, text, flags=re.I)
    text = re.sub(regex % "provider number", replacement, text, flags=re.I)

    # NAME
    text = re.sub(regex % "name", replacement, text, flags=re.I)
    text = re.sub(regex % "dictator", replacement, text, flags=re.I)
    text = re.sub(regex % "attending", replacement, text, flags=re.I)

    # HOSPITAL
    text = re.sub(regex % "hospital", replacement, text, flags=re.I)

    # LOC
    text = re.sub(regex % "location", replacement, text, flags=re.I)
    text = re.sub(regex % "address", replacement, text, flags=re.I)
    text = re.sub(regex % "country", replacement, text, flags=re.I)
    text = re.sub(regex % "state", replacement, text, flags=re.I)
    text = re.sub(regex % "university", replacement, text, flags=re.I)

    # DATE
    text = re.sub(regex % "year", replacement, text, flags=re.I)
    text = re.sub(regex % "month", replacement, text, flags=re.I)
    text = re.sub(regex % "date", replacement, text, flags=re.I)
    text = re.sub(r"\[?\*\*[0-9]{0,2}/[0-9]{0,4}\*\*\]?", replacement, text, flags=re.I)  # e.g. 03/1990
    text = re.sub(r"\[?\*\*[0-9]{0,4}\*\*\]?", replacement, text, flags=re.I)  # e.g. 1991
    text = re.sub(r"\[?\*\*(?:[0-9]{0,4}-)?[0-9]{0,2}-[0-9]{0,2}\*\*\]?", replacement, text, flags=re.I)

    # CLIP
    text = re.sub(r"\[?\*\*.*Clip Number.*\*\*\]?", replacement, text, flags=re.I)

    # HOLIDAY
    text = re.sub(r"\[?\*\*.*Holiday.*\*\*\]?", replacement, text, flags=re.I)

    # COMPANY
    text = re.sub(r"\[?\*\*.*Company.*\*\*\]?", replacement, text, flags=re.I)

    # JOB
    text = re.sub(r"\[?\*\*.*Job Number.*\*\*\]?", replacement, text, flags=re.I)

    # UNIT_NUM
    text = re.sub(r"\[?\*\*.*Unit Number.*\*\*\]?", replacement, text, flags=re.I)

    # URL
    text = re.sub(r"\[?\*\*.*url.*\*\*\]?", replacement, text, flags=re.I)

    # OTHER
    text = re.sub(r"\[?\*\*.*\d+.*\*\*\]?", replacement, text, flags=re.I)
    text = re.sub(r"\[?\*\* +\*\*\]?", replacement, text, flags=re.I)

    return text
