import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    data_list = []
    
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';')

        for row in csv_reader:
            Data_date = row['Date']
            Data_time = row['Time']
            Data_CO = row['CO(GT)'] 
            Data_CO = Data_CO.replace(',', '.')

            # Remove the irrelevant data
            if Data_CO == '-200':
                Data_CO = 'NA'

            data_item = {
                "user": f"What is the value of {Data_date} at {Data_time}?",
                "assistant": Data_CO
            }
            data_list.append(data_item)
    
    # Write to JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)

csv_file_path = r"C:\Users\vaibh\OneDrive\Documents\UCLA_Courses\M202A - Embedded Systems\LLMs_ReadingaMaterial\FineTuning\Dataset\AirQualityUCI.csv"
#csv_file_path = r"C:\Users\vaibh\OneDrive\Documents\UCLA_Courses\M202A - Embedded Systems\LLMs_ReadingaMaterial\FineTuning\input.csv"
json_file_path = r"C:\Users\vaibh\OneDrive\Documents\UCLA_Courses\M202A - Embedded Systems\LLMs_ReadingaMaterial\FineTuning\Output.json"

csv_to_json(csv_file_path, json_file_path)
