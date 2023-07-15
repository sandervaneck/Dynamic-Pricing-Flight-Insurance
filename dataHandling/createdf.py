from parser import from_comma_separated_amount, parse_csv, to_date, parse_date_string2
import pandas as pd

def create_weathers(weathers, weather_paths, states):
  for index, weather_path in enumerate(weather_paths):
    data = parse_csv(weather_path)
    entries = []
    for _, row in data.iterrows():
      state = row["name"]
      if state not in states:
        continue
      temp = from_comma_separated_amount(row["temp"])
      dew = from_comma_separated_amount(row["dew"])
      humidity = from_comma_separated_amount(row["humidity"])
      precip = from_comma_separated_amount(row["precip"])
      windgust = from_comma_separated_amount(row["windgust"])
      windspeed = from_comma_separated_amount(row["windspeed"])
      visibility = from_comma_separated_amount(row["visibility"])
      d = row["datetime"]
      date = to_date(d)
      entry = {
        "date": date,
        "state": state,
        "temp": temp,
        "dew": dew,
        "humidity": humidity,
        "precip": precip,
        "windgust": windgust,
        "windspeed": windspeed,
        "visibility": visibility
      }
      entries.append(entry)
  weathers.extend(entries)
  return weathers


def parse_data(flight_data_paths, summarized_data, states, weather_paths, weathers, variables):
    weathers = create_weathers(weathers, weather_paths, states)
    for path in flight_data_paths:
        data = parse_csv(path)
        data = data.sample(n=int(len(data) * 0.01), replace=False, random_state=42)

        data = data[data['ORIGIN_STATE_NM'].isin(states)]
        entries = []
        for _, row in data.iterrows():
            state = row["ORIGIN_STATE_NM"]
            if state not in states:
                continue
            d = row["FL_DATE"]
            parsed_d = d.split("2022", 1)[0].strip() + "2022"
            date = parse_date_string2(parsed_d)
            carrier = row["OP_UNIQUE_CARRIER"]
            origin = row["ORIGIN"]
            destination = row["DEST"]
            dep_time = from_comma_separated_amount(row["CRS_DEP_TIME"])
            delay = from_comma_separated_amount(row["ARR_DELAY"])
            cancelled = from_comma_separated_amount(row["CANCELLED"])
            diverted = from_comma_separated_amount(row["DIVERTED"])
            flights = from_comma_separated_amount(row["FLIGHTS"])
            distance = from_comma_separated_amount(row["DISTANCE"])
            newEntry = {
                "date": date,
                "carrier": carrier,
                "origin": origin,
                "state": state,
                "destination": destination,
                "depTime": dep_time,
                "delay": delay,
                "cancelled": cancelled,
                "diverted": diverted,
                "flights": flights,
                "distance": distance
            }
            entries.append(newEntry)
            filtered_entries = entries
            summarized_entries = []
            for row in filtered_entries:
                w = next((w for w in weathers if w["state"] == row["state"] and w["date"] == row["date"]), None)
                if w is not None:
                    summarized_entry = {
                        "carrier": row["carrier"],
                        "distance": row["distance"],
                        "origin": row["origin"],
                        "state": row["state"],
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
                        "refund": 1.0 if (row["delay"]) >= 120 or (row["cancelled"]) == 1.0 else 0.0
                    }
                    summarized_entries.append(summarized_entry)
            summarized_data.extend(summarized_entries)
    summarized_data = pd.DataFrame(summarized_data, columns=variables)
    summarized_data['delay'] = pd.to_numeric(summarized_data['delay'])
    return summarized_data
