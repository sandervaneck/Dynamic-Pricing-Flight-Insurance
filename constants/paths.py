from openpyxl import Workbook

weather_paths = [
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/New York 2022-01-01 to 2022-12-31.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/Hawaii 2022-01-01 to 2022-12-31.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/Colorado 2022-01-01 to 2022-12-31.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/Florida 2022-01-01 to 2022-12-31.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/arizona 2022-01-01 to 2022-12-31.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/Connecticut 2022-01-01 to 2022-12-31.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/Delaware 2022-01-01 to 2022-12-31.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/Pennsylvania 2022-01-01 to 2022-12-31.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/Seattle 2022-01-01 to 2022-12-31.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/Georgia 2022-01-01 to 2022-12-31.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/Texas 2022-01-01 to 2022-12-31.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/Tennessee 2022-01-01 to 2022-12-31.csv"
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/Los Angeles 2022-01-01 to 2022-12-31.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/Illinois 2022-01-01 to 2022-12-31.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/Indiana 2022-01-01 to 2022-12-31.csv"
]
flight_data_paths = [
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/FlightData/012022Input.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/FlightData/022022Input.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/FlightData/032022Input.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/FlightData/042022Input.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/FlightData/052022Input.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/FlightData/062022Input.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/FlightData/072022Input.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/FlightData/082022Input.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/FlightData/092022Input.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/FlightData/102022Input.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/FlightData/112022Input.csv",
        "/Users/sander/Library/CloudStorage/GoogleDrive-mychefsbase@gmail.com/My Drive/Mack Backup/Thesis/FlightData/122022Input.csv"
    ]
headers = ['date', 'distance', 'temp', 'dew', 'humidity', 'precip', 'windgust', 'windspeed', 'visibility','delay', 'refund', 'carrier', 'origin', 'state']
cat_var = ['distance', 'carrier', 'origin', 'state']
num_var = ['scaled_origin_score', 'scaled_carrier_score', 'distance', 'temp', 'dew', 'humidity', 'precip','windgust', 'windspeed', 'visibility']
df_var = ['refund', 'scaled_origin_score', 'scaled_carrier_score', 'distance', 'temp', 'dew', 'humidity', 'precip','windgust', 'windspeed', 'visibility']
factors = ['scaled_origin_score', 'scaled_carrier_score', 'distance', 'windspeed', 'visibility', 'precip']
rfc_factors = ['scaled_origin_score', 'scaled_carrier_score', 'distance', 'windspeed', 'visibility', 'precip', 'temp', 'dew', 'humidity']


# , 'temp', 'dew', 'humidity', 'precip','windgust', 'windspeed', 'visibility']

workbookDf = Workbook()
dataFile = "Data.xlsx"
workbookTrainDf = Workbook()
randomForestFile = "RandomForest.xlsx"
workbookResultsDF = Workbook()
resultFile = "Result.xlsx"