import pandas as pd
from datetime import datetime

def parse_csv(file_path):
    with open(file_path, 'r') as file:
        # Read a portion of the file to determine the delimiter
        sample_data = file.read(1024)  # You can adjust the number of bytes to read

        # Check if the ';' delimiter is present in the sample data
        delimiter = ';' if ';' in sample_data else ','

    return pd.read_csv(file_path, delimiter=delimiter)

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
        date_number = int(d)
        date_obj = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + date_number - 2)
        return date_obj.date()
    except (ValueError, OverflowError):
        pass

    try:
        date_format = '%Y-%m-%d'
        date = datetime.strptime(d, date_format).date()
    except ValueError:
        date_format = '%d/%m/%Y'
        date = datetime.strptime(d, date_format).date()

    return date

def parse_date_string2(date_string):
    date_formats = ["%m/%d/%Y", "%m/%d/%y", "%-m/%-d/%Y"]  # Define the desired date formats

    try:
        date_number = int(date_string)
        date_obj = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + date_number - 2)
        return date_obj.date()
    except (ValueError, OverflowError):
        pass

    for date_format in date_formats:
        try:
            return datetime.strptime(date_string, date_format).date()
        except ValueError:
            pass

    return None