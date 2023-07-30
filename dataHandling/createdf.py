from constants.paths import flight_data_paths
from dataHandling.parser import from_comma_separated_amount, parse_csv, to_date, parse_date_string2
import pandas as pd

def create_weathers(weathers, weather_paths):
    for index, weather_path in enumerate(weather_paths):
        data = parse_csv(weather_path)
        entries = []
        for _, row in data.iterrows():
            origin = row["name"]
            if origin == "new york city":
                origin = "JFK"
            elif origin == "orlando":
                origin = "MCO"
            elif origin == "denver":
                origin = "DEN"
            elif origin == "chicago":
                origin = "ORD"
            elif origin == "miami":
                origin = "MIA"
            elif origin == "Atlanta":
                origin = "ATL"
            elif origin == "Dallas":
                origin = "DFW"
            elif origin == "Los Angeles":
                origin = "LAX"
            elif origin == "Las vegas":
                origin = "LAS"
            else:
                origin = ""
            if origin not in ['ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'JFK', 'LAS', 'MCO', 'MIA']:
                continue
            temp = from_comma_separated_amount(row["temp"])
            dew = from_comma_separated_amount(row["dew"])
            humidity = from_comma_separated_amount(row["humidity"])
            precip = from_comma_separated_amount(row["precip"])
            windgust = from_comma_separated_amount(row["windgust"])
            windspeed = from_comma_separated_amount(row["windspeed"])
            visibility = from_comma_separated_amount(row["visibility"])
            cloudcover = from_comma_separated_amount(row["cloudcover"])
            snowdepth = from_comma_separated_amount(row["snowdepth"])
            sealevelpressure = from_comma_separated_amount(row["sealevelpressure"])
            d = row["datetime"]
            date = to_date(d)
            entry = {
                "date": date,
                "origin": origin,
                "temp": temp,
                "dew": dew,
                "humidity": humidity,
                "precip": precip,
                "windgust": windgust,
                "windspeed": windspeed,
                "visibility": visibility,
                "cloudcover": cloudcover,
                "snowdepth": snowdepth,
                "sealevelpressure": sealevelpressure,
                # "severerisk": severerisk
            }
            entries.append(entry)
        weathers.extend(entries)  # Extend 'weathers' after processing all rows in the current weather file
    return weathers

def create_data(flight_data_paths):
    entries = []
    for path in flight_data_paths:
        data = parse_csv(path)
        #For testing purposes
        data = data.sample(n=int(len(data) *0.1), replace=False, random_state=42)
        for _, row in data.iterrows():
            origin = row["ORIGIN"]
            if origin not in ['ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'JFK', 'LAS', 'MCO', 'MIA']:
                continue

            d = row["FL_DATE"]
            parsed_d = d.split("2022", 1)[0].strip() + "2022"
            date = parse_date_string2(parsed_d)
            weekday = pd.to_datetime(date).weekday()

            carrier = row["OP_UNIQUE_CARRIER"]
            destination = row["DEST"]
            dep_time = from_comma_separated_amount(row["CRS_DEP_TIME"])
            delay = from_comma_separated_amount(row["ARR_DELAY"])
            cancelled = from_comma_separated_amount(row["CANCELLED"])
            diverted = from_comma_separated_amount(row["DIVERTED"])
            flights = from_comma_separated_amount(row["FLIGHTS"])
            distance = from_comma_separated_amount(row["DISTANCE"])
            CRS_DEP_TIME = from_comma_separated_amount(row["CRS_DEP_TIME"])
            if 0 <= CRS_DEP_TIME < 600:
                time = 1
            elif 600 <= CRS_DEP_TIME < 1200:
                time = 2
            elif 1200 <= CRS_DEP_TIME < 1800:
                time = 3
            elif 1800 <= CRS_DEP_TIME < 2400:
                time = 4
            else:
                time = 0
            newEntry = {
                "date": date,
                "carrier": carrier,
                "origin": origin,
                "destination": destination,
                "depTime": dep_time,
                "delay": delay,
                "cancelled": cancelled,
                "diverted": diverted,
                "flights": flights,
                "distance": distance,
                'weekday': weekday,
                "time": time,
            }
            entries.append(newEntry)
    df = pd.DataFrame(entries)
    return df


variables = ['distance', "carrier", 'origin', 'date', 'cancelled', 'delay', 'temp', 'dew', 'humidity', 'precip','windgust', 'windspeed', 'visibility', 'cloudcover', 'snowdepth', 'sealevelpressure','weekday', 'time', 'refund']

def parse_data(summarized_data, weather_paths, weathers):
    data = create_data(flight_data_paths)
    weathers = create_weathers(weathers, weather_paths)
    for _, row in data.iterrows():
        w = next((w for w in weathers if w["origin"] == row["origin"] and w["date"] == row["date"]), None)
        if w is not None:
            summarized_entry = {
                "carrier": row["carrier"],
                "distance": row["distance"],
                "origin": row["origin"],
                "date": row["date"],
                "cancelled": row["cancelled"],
                "delay": row["delay"],
                "temp": w["temp"],
                "dew": w["dew"],
                "humidity": w["humidity"],
                "precip": w["precip"],
                "windgust": w["windgust"],
                "windspeed": w["windspeed"],
                "visibility": w["visibility"],
                "cloudcover": w["cloudcover"],
                "snowdepth": w["snowdepth"],
                "sealevelpressure": w["sealevelpressure"],
                "weekday": row["weekday"],
                "time": row["time"],
                "refund": 1.0 if (row["delay"]) >= 120 or (row["cancelled"]) == 1.0 else 0.0
            }
        summarized_data.append(summarized_entry)
    summarized_data = pd.DataFrame(summarized_data, columns=variables)
    summarized_data['delay'] = pd.to_numeric(summarized_data['delay'])

    weekday_map = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }
    summarized_data["windgust"].fillna(0, inplace=True)

    # Use the mapping to replace the weekday numbers with weekday names
    summarized_data['weekday'] = summarized_data['weekday'].map(weekday_map)

    # Convert the 'weekday' column to a categorical variable
    summarized_data['weekday'] = pd.Categorical(summarized_data['weekday'], categories=weekday_map.values(), ordered=True)

    return summarized_data
