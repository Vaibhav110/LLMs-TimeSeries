import csv
import json

def Time_output(Data_date):
    # Extract hours, minutes, and seconds
    hours, minutes, seconds = Data_date.split(".")

    # Convert hours to integer for comparison
    return hours

def csv_to_json(csv_file_path, json_file_path):
    data_list = []
    
    # Read the CSV file
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';')

        # first_row = next(csv_reader)
        # print(first_row)

        # Convert the CSV reader object to a list
        csv_list = list(csv_reader)
        #print(csv_list)

        prev_date = csv_list[0]['Date']
        Data_date = csv_list[0]['Date']
        no_of_days = 0
        count_n = 0
        count_m = 0
        avg_morning_CO = 0
        avg_night_CO = 0
        four_dates = []
        four_dates_mdata = []
        four_dates_ndata = []
        four_dates_count = 0

        # Iterate through each row in the CSV
        for row in csv_list:

            Data_date = row['Date']
            Data_time = int(Time_output( row['Time'] ) )
            Data_CO = row['CO(GT)'] 
            Data_CO = Data_CO.replace(',', '.')

            #print("Date = "+ Data_date + " Time = " + str(Data_time))

            # Count the number of daya
            if prev_date != Data_date:
                four_dates.append(prev_date)
                four_dates_count = four_dates_count + 1

                if count_m != 0:
                    avg_morning_CO = round(avg_morning_CO / count_m , 2)
                    four_dates_mdata.append(avg_morning_CO)
                else:
                    four_dates_mdata.append('NA')

                if count_n != 0:
                    avg_night_CO = round( avg_night_CO / count_n , 2)
                    four_dates_ndata.append(avg_night_CO)
                else:
                    four_dates_ndata.append('NA')

                no_of_days = no_of_days + 1
                prev_date = Data_date
                count_n = 0
                count_m = 0
                avg_morning_CO = 0
                avg_night_CO = 0
                if four_dates_count == 5:
                    print(four_dates)
                    print(four_dates_mdata)
                    print(four_dates_ndata)
                    data_item = {
                        "user": f"Average day data of CO on {four_dates[0]} is {four_dates_mdata[0]}, on {four_dates[1]} is {four_dates_mdata[1]}, on {four_dates[2]} is {four_dates_mdata[2]}, on {four_dates[3]} is {four_dates_mdata[3]}. What is average day data the next day? Just tell the answer",
                        "assistant": f"{four_dates_mdata[4]}"
                    }
                    data_list.append(data_item)

                    data_item = {
                        "user": f"Average night data of CO on {four_dates[0]} is {four_dates_ndata[0]}, on {four_dates[1]} is {four_dates_ndata[1]}, on {four_dates[2]} is {four_dates_ndata[2]}, on {four_dates[3]} is {four_dates_ndata[3]}. What is average night data the next night? Just tell the answer",
                        "assistant": f"{four_dates_ndata[4]}"
                    }
                    data_list.append(data_item)

                    four_dates = []
                    four_dates_mdata = []
                    four_dates_ndata = []
                    four_dates_count = 0

            # Remove the irrelevant data
            if Data_CO == '-200':
                Data_CO = 'NA'
            else:
                if Data_time >= 6 and Data_time < 18:
                    avg_morning_CO = avg_morning_CO + float(Data_CO)
                    count_m = count_m + 1
                else:
                    avg_night_CO = avg_night_CO + float(Data_CO)
                    count_n = count_n + 1

    print('No of days = ' + str(no_of_days))

    # Write to JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)

# Replace these paths with the appropriate file paths
csv_file_path = r"/home/vaibhav/LLMs-TimeSeries/data/AirQualityUCI.csv"
json_file_path = r"/home/vaibhav/LLMs-TimeSeries/software/Output.json"
# Call the function
csv_to_json(csv_file_path, json_file_path)
