import csv
import json
import numpy as np
import random
import pandas as pd
import datetime
from MTF_GAF import Scalogram_conv_out, MtF_Conv_save, Spectrogram_conv_save

def get_day_of_week(date_string):
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

        count = 0
        Data_Size=6

        # Iterate through each row in the CSV
        for row in csv_list:

            Data_date = row["Date"]
            Data_season, Data_year = Season_output( Data_date )
            Data_day = get_day_of_week(Data_date)
            Data_NO2 = int(float(row["Daily Max 1-hour NO2 Concentration"])*10)

            img_path = []
            print("Date = "+ Data_date + " NO2= " + str(Data_NO2) + "DAY = " + Data_day + " Season = " + Data_season)

            four_dates_count = four_dates_count + 1

            no_of_days = no_of_days + 1

            if four_dates_count == (Data_Size+1):
                img_path.append(f"{img_path_prefix[0]}" + str(count) + '.jpg')
                img_path.append(f"{img_path_prefix[1]}" + str(count) + '.jpg')
                img_path.append(f"{img_path_prefix[2]}" + str(count) + '.jpg')

                data_item = {
                    "id":f"id_{count}",
                    "image_path_scalogram": f"{img_path[0]}",
                    "query_scalogram": f"Value on Nitrogen Dioxide in San Francisco on six consecutive days from {Combined_NO2_days[0]} to {Combined_NO2_days[Data_Size-1]} during the {Data_season} {Data_year} season is {", ".join(str(s) for s in Combined_NO2_data)}. Analyze the provided {img_type_scalogram} of Nitrogen Dioxide for six days and estimate the expected Nitrogen Dioxide value for the subsequent day that is {Data_day}.",
                    # "query_scalogram": f"Value on Nitrogen Dioxide on ten consecutive days is {", ".join(str(s) for s in Combined_NO2_data)}. Analyze the provided {img_type_scalogram} of Nitrogen Dioxide for ten days and estimate the expected average Nitrogen Dioxide value for the subsequent day.",
                    "image_path_mtf": f"{img_path[1]}",
                    "query_mtf": f" Value on Nitrogen Dioxide in San Francisco on six consecutive days from {Combined_NO2_days[0]} to {Combined_NO2_days[Data_Size-1]} during the {Data_season} {Data_year} season is {", ".join(str(s) for s in Combined_NO2_data)}. Analyze the provided {img_type_mtf} of Nitrogen Dioxide for six days and estimate the expected Nitrogen Dioxide value for the subsequent day that is {Data_day}.",
                    # "query_mtf": f"Value on Nitrogen Dioxide on ten consecutive days is {", ".join(str(s) for s in Combined_NO2_data)}. Analyze the provided {img_type_mtf} of Nitrogen Dioxide for ten days and estimate the expected average Nitrogen Dioxide value for the subsequent day.",
                    "image_path_spectrogram": f"{img_path[2]}",
                    "query_spectrogram": f"Value on Nitrogen Dioxide in San Francisco on six consecutive days from {Combined_NO2_days[0]} to {Combined_NO2_days[Data_Size-1]} during the {Data_season} {Data_year} season is {", ".join(str(s) for s in Combined_NO2_data)}.Analyze the provided {img_type_spectrogram} of Nitrogen Dioxide for six days and estimate the expected Nitrogen Dioxide value for the subsequent day that is {Data_day}.",
                    # "query_spectrogram": f"Value on Nitrogen Dioxide on ten consecutive days is {", ".join(str(s) for s in Combined_NO2_data)}. Analyze the provided {img_type_spectrogram} of Nitrogen Dioxide for ten days and estimate the expected average Nitrogen Dioxide value for the subsequent day.",                 
                    "answers": f"{Data_NO2}"
                }
                json_list.append(data_item)
                four_dates_count = 0
                count = count + 1
                if IS_SCALOGRAM:
                    Scalogram_conv_out(Combined_NO2_data, img_path[0])
                if IS_MTF:
                    data = np.array([Combined_NO2_data])
                    MtF_Conv_save(data, img_path[1])
                if IS_SPECTROGRAM:
                    data = np.array(Combined_NO2_data)
                    Spectrogram_conv_save(data, img_path[2])
                Combined_NO2_data = []
                Combined_NO2_days = []
            else:
                Combined_NO2_data.append(Data_NO2)
                Combined_NO2_days.append(Data_day)

    print('No of days = ' + str(no_of_days))
    return json_list



# ---------------------------------------------------------------------------------

#################################################3
IS_SCALOGRAM = True
IS_MTF = True
IS_SPECTROGRAM = True
#####################################################
MORE_CONTEXT_QUERY = True
################################################
img_path_prefix = []
if IS_SCALOGRAM:
    # img_path_prefix = "Dataset/Images_Scalogram/Data_"
    img_path_prefix.append("Dataset_SF_6/Images_Scalogram/Data_")
    img_type_scalogram = 'Scalogram'
if IS_MTF:
    # img_path_prefix = "Dataset/Images_MTF/Data_"
    img_path_prefix.append("Dataset_SF_6/Images_MTF/Data_")
    img_type_mtf = 'Markov Transition Field'
if IS_SPECTROGRAM:
    # img_path_prefix = "Dataset/Images_Spectrogram/Data_"
    img_path_prefix.append("Dataset_SF_6/Images_Spectrogram/Data_")
    img_type_spectrogram = 'Spectrogram'



final_csv_path = r"C:\Users\vaibh\OneDrive\Documents\UCLA_Courses\M202A - Embedded Systems\LLMs_ReadingaMaterial\Dataset_v2\AirQuality_NO2_2023_SF.csv"

# Call the function
raw_data = csv_to_json(final_csv_path)

json_file_path_train = r"C:\Users\vaibh\OneDrive\Documents\UCLA_Courses\M202A - Embedded Systems\LLMs_ReadingaMaterial\Dataset_v2\Output_dataset_image_test_v3.json"




# Write to JSON file
with open(json_file_path_train, 'w') as json_file:
    json.dump(raw_data, json_file, indent=4)
