import csv
import json
from Dataset_play.Replacing_200_script import Remove_200_data, read_the_file, save_the_file

def Time_output(Data_date):
    # Extract hours, minutes, and seconds
    hours, minutes, seconds = Data_date.split(".")

    # Convert hours to integer for comparison
    return hours

def csv_to_json(csv_file_path):
    data_list = []

    output_dataset_list = []
    
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
        avg_morning_CO = []
        avg_night_CO = []
        four_dates = []
        four_dates_mdata = []
        four_dates_ndata = []
        four_dates_count = 0

        # Iterate through each row in the CSV
        for row in csv_list:

            Data_date = row['Date']
            Data_time = int(Time_output( row['Time'] ) )
            Data_CO = row['PT08.S4(NO2)'] 
            Data_CO = Data_CO.replace(',', '.')

            #print("Date = "+ Data_date + " Time = " + str(Data_time))

            # Count the number of daya
            if prev_date != Data_date:
                four_dates.append(prev_date)
                four_dates_count = four_dates_count + 1

                four_dates_mdata.append(avg_morning_CO)
                four_dates_ndata.append(avg_night_CO)

                no_of_days = no_of_days + 1
                prev_date = Data_date
                avg_morning_CO = []
                avg_night_CO = []
                if four_dates_count == 5:
                    # print(four_dates)
                    # print(four_dates_mdata)
                    # print(four_dates_ndata)
                    output_dataset_list.append(four_dates_mdata)
                    output_dataset_list.append(four_dates_ndata)
                    # data_item = {
                    #     "user": f"Average day data of CO on {four_dates[0]} is {four_dates_mdata[0]}, on {four_dates[1]} is {four_dates_mdata[1]}, on {four_dates[2]} is {four_dates_mdata[2]}, on {four_dates[3]} is {four_dates_mdata[3]}. What is average day data the next day? Just tell the answer",
                    #     "assistant": f"{four_dates_mdata[4]}"
                    # }
                    # data_list.append(data_item)

                    # data_item = {
                    #     "user": f"Average night data of CO on {four_dates[0]} is {four_dates_ndata[0]}, on {four_dates[1]} is {four_dates_ndata[1]}, on {four_dates[2]} is {four_dates_ndata[2]}, on {four_dates[3]} is {four_dates_ndata[3]}. What is average night data the next night? Just tell the answer",
                    #     "assistant": f"{four_dates_ndata[4]}"
                    # }
                    # data_list.append(data_item)

                    four_dates = []
                    four_dates_mdata = []
                    four_dates_ndata = []
                    four_dates_count = 0

            # Remove the irrelevant data
            if Data_time >= 6 and Data_time < 18:
                # avg_morning_CO = avg_morning_CO + float(Data_CO)
                avg_morning_CO.append(Data_CO)
            else:
                # avg_night_CO = avg_night_CO + float(Data_CO)
                avg_night_CO.append(Data_CO)

    print('No of days = ' + str(no_of_days))
    return output_dataset_list


# Replace these paths with the appropriate file paths
csv_file_path = r"C:\Users\vaibh\OneDrive\Documents\UCLA_Courses\M202A - Embedded Systems\LLMs_ReadingaMaterial\FineTuning\Dataset\AirQualityUCI_mod.csv"
#csv_file_path = r"C:\Users\vaibh\OneDrive\Documents\UCLA_Courses\M202A - Embedded Systems\LLMs_ReadingaMaterial\FineTuning\input.csv"

# Call the function
raw_data = csv_to_json(csv_file_path)
save_the_file(raw_data, 'Raw_data.txt')

output = Remove_200_data(raw_data)
save_the_file(output, 'Raw_data_wo_200.txt')


print(output[0])