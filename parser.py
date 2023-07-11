import pandas as pd
from datetime import datetime

def parse_csv(file_path):
    return pd.read_csv(file_path, delimiter=';')

def from_comma_separated_amount(amount):
    if isinstance(amount, float):
        return amount
    elif isinstance(amount, int):
        amount = str(amount)
    elif amount == "":
        return 0.0
    return float(amount.replace(",", "."))

def to_date(d):
    try:
        date_format = '%Y-%m-%d'
        date = datetime.strptime(d, date_format).date()
    except ValueError:
        date_format = '%d/%m/%Y'
        date = datetime.strptime(d, date_format).date()

    return date

def parse_date_string2(date_string):
    date_formats = ["%m/%d/%Y", "%m/%d/%y", "%-m/%-d/%Y"]  # Define the desired date formats

    for date_format in date_formats:
        try:
            return datetime.strptime(date_string, date_format).date()
        except ValueError:
            pass

    return None  # Parsing failed with all formats