import csv
import json
import numpy as np
import random
import pandas as pd
import datetime

def get_day_of_week(date_string):
  """Returns the day of the week for a given date string in MM-DD-YYYY format.

  Args:
    date_string: The date string in the format 'MM-DD-YYYY'.

  Returns:
    The day of the week as a string (e.g., 'Monday', 'Tuesday').
  """

  date_object = datetime.datetime.strptime(date_string, '%m/%d/%Y')
  day_of_week = date_object.strftime('%A')
  return day_of_week



def Season_output(Data_date):
    # Extract month, date and year
    month, date, year = Data_date.split("/")
    Seasons_list = ['Winter', 'Spring', 'Summer', 'Fall']
    month = int(month)
    if month < 4:
      return Seasons_list[0], year
    elif month < 7:
      return Seasons_list[1], year
    elif month < 10:
      return Seasons_list[2], year
    else:
      return Seasons_list[3], year
    # Convert hours to integer for comparison
    # return date

def csv_to_json(csv_file_path):
    json_list = []
    csv_list = []

    # Read the CSV file
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')

        # first_row = next(csv_reader)
        # print(first_row)

        # Convert the CSV reader object to a list
        csv_list = list(csv_reader)
        # print(csv_list[0])
        # print(csv_list[363])
        #print(csv_list)

        no_of_days = 0

        Combined_NO2_data = []
        Combined_NO2_days = []
        four_dates_count = 0

        Data_Size = 7

        # Iterate through each row in the CSV
        for row in csv_list:

            Data_date = row["Date"]
            Data_season, Data_year = Season_output( Data_date )
            Data_day = get_day_of_week(Data_date)
            Data_NO2 = int(float(row["Daily Max 1-hour NO2 Concentration"])*10)

            print("Date = "+ Data_date + " NO2= " + str(Data_NO2) + "DAY = " + Data_day + " Season = " + Data_season)

            four_dates_count = four_dates_count + 1

            no_of_days = no_of_days + 1

            if four_dates_count == (Data_Size+1):

                data_item = {
                    # "user": f" Value on Nitrogen Dioxide in Los Angeles on seven consecutive days {", ".join(str(s) for s in Combined_NO2_days)} during the {Data_season} {Data_year} season is {", ".join(str(s) for s in Combined_NO2_data)}. Estimate the expected Nitrogen Dioxide value for the subsequent day that is {Data_day}.",
                    "user": f"Value on Nitrogen Dioxide in Los Angeles on seven consecutive days from {Combined_NO2_days[0]} to {Combined_NO2_days[Data_Size-1]} during the {Data_season} {Data_year} season is {", ".join(str(s) for s in Combined_NO2_data)}. Estimate the expected Nitrogen Dioxide value for the subsequent day that is {Data_day}.",
                    "assistant": f"{Data_NO2}"
                }
                json_list.append(data_item)
                four_dates_count = 0
                Combined_NO2_data = []
                Combined_NO2_days = []
            else:
                Combined_NO2_data.append(Data_NO2)
                Combined_NO2_days.append(Data_day)

    print('No of days = ' + str(no_of_days))
    return json_list


# Replace these paths with the appropriate file paths
csv_file_path1 = r"C:\Users\vaibh\OneDrive\Documents\UCLA_Courses\M202A - Embedded Systems\LLMs_ReadingaMaterial\Dataset_v2\AirQuality_NO2_2016_LA.csv"
csv_file_path2 = r"C:\Users\vaibh\OneDrive\Documents\UCLA_Courses\M202A - Embedded Systems\LLMs_ReadingaMaterial\Dataset_v2\AirQuality_NO2_2017_LA.csv"
csv_file_path3 = r"C:\Users\vaibh\OneDrive\Documents\UCLA_Courses\M202A - Embedded Systems\LLMs_ReadingaMaterial\Dataset_v2\AirQuality_NO2_2018_LA.csv"
csv_file_path4 = r"C:\Users\vaibh\OneDrive\Documents\UCLA_Courses\M202A - Embedded Systems\LLMs_ReadingaMaterial\Dataset_v2\AirQuality_NO2_2019_LA.csv"

final_csv_path = r"C:\Users\vaibh\OneDrive\Documents\UCLA_Courses\M202A - Embedded Systems\LLMs_ReadingaMaterial\Dataset_v2\AirQuality_combined_csv_LA.csv"

# Read both CSV files
df1 = pd.read_csv(csv_file_path1)
df2 = pd.read_csv(csv_file_path2)
df3 = pd.read_csv(csv_file_path3)
df4 = pd.read_csv(csv_file_path4)

# Concatenate the DataFrames
df_combined = pd.concat(    [  pd.concat([df1, df2], ignore_index=True), df3], ignore_index=True)

df_combined = pd.concat([df_combined, df4], ignore_index=True)

# Write the combined DataFrame to a new CSV file
df_combined.to_csv(final_csv_path, index=False)

# Call the function
raw_data = csv_to_json(final_csv_path)

# json_file_path_train = r"C:\Users\vaibh\OneDrive\Documents\UCLA_Courses\M202A - Embedded Systems\LLMs_ReadingaMaterial\Dataset_v2\Output_dataset_text.json"
json_file_path_train = r"C:\Users\vaibh\OneDrive\Documents\UCLA_Courses\M202A - Embedded Systems\LLMs_ReadingaMaterial\Dataset_v2\Output_dataset_text_mContext.json"


# Write to JSON file
with open(json_file_path_train, 'w') as json_file:
    json.dump(raw_data, json_file, indent=4)
