""" Python Data Type Parser

"""


# region Imported Dependencies
import re
from datetime import datetime
# endregion Imported Dependencies


def parser(text):
    if text.startswith('{') and text.endswith('}'):
        kwargs = {}
        key_value_pairs = text[1:-1].split(',')
        for pair in key_value_pairs:
            if len(pair) == 0:
                continue
            key, value = pair.split(':', 1)
            key = key.strip()
            value = value.strip()
            kwargs[key] = parser(value)  # Recursively parse the value
        return kwargs

    if text.startswith('[') and text.endswith(']'):
        args = []
        items = text[1:-1].split(',')
        for item in items:
            if len(item) == 0:
                continue
            args.append(parser(item.strip()))
        return args

    # Check for integers
    if re.match(r'^-?\d+$', text):
        return int(text)

    # Check for floats (including the "1e-5" format)
    if re.match(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$', text):
        return float(text)

    # Check for boolean values
    if text.lower() == 'true':
        return True
    elif text.lower() == 'false':
        return False

    # Check for dates (you can define your date format patterns)
    date_formats = ["%Y-%m-%d", "%m/%d/%Y"]
    for date_format in date_formats:
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            pass

    # If none of the above, treat it as a string
    return text
