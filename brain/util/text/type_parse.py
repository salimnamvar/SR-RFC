""" Python Data Type Parser

"""


# region Imported Dependencies
import re
from datetime import datetime
# endregion Imported Dependencies


def parser(text):
    # Check for integers
    if re.match(r'^-?\d+$', text):
        return int(text)

    # Check for floats
    if re.match(r'^-?\d+\.\d+$', text):
        return float(text)

    # Check for dates (you can define your date format patterns)
    date_formats = ["%Y-%m-%d", "%m/%d/%Y"]
    for date_format in date_formats:
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            pass

    # Check for boolean values
    if text.lower() == 'true':
        return True
    elif text.lower() == 'false':
        return False

    # Check for lists and tuples (assuming comma-separated values)
    if ',' in text:
        items = [item.strip() for item in text.split(',')]
        if '(' in text and ')' in text:
            return tuple(items)
        else:
            return list(items)

    # If none of the above, treat it as a string
    return text
