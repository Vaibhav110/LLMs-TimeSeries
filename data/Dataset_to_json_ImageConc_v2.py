import csv
import json
from Dataset_play.Replacing_200_script import Remove_200_data, read_the_file, save_the_file
from MTF_GAF import Scalogram_conv_out

def Time_output(Data_date):
    # Extract hours, minutes, and seconds
    hours, minutes, seconds = Data_date.split(".")

    # Convert hours to integer for comparison
    return hours

def csv_to_dataset(csv_file_path):
    output_dataset_list = []
    Prediction_date = []
    
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
        Moring_data = []
        Night_data = []
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

                four_dates_mdata.append(Moring_data)
                four_dates_ndata.append(Night_data)

                no_of_days = no_of_days + 1
                prev_date = Data_date
                # Moring_data = []
                # Night_data = []
                if four_dates_count == 5:
                    # print(four_dates)
                    # print(four_dates_mdata)
                    # print(four_dates_ndata)
                    # Morning data contains the day data (12hrs each)  for 5 days. Splitting 4 days for the image and last day for expected output
                    output_dataset_list.append(Moring_data)
                    output_dataset_list.append(Night_data)
                    Prediction_date.append(four_dates[4])
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
                    Moring_data = []
                    Night_data = []

            # Remove the irrelevant data
            if Data_time >= 6 and Data_time < 18:
                # Moring_data = Moring_data + float(Data_CO)
                Moring_data.append(Data_CO)
            else:
                # Night_data = Night_data + float(Data_CO)
                Night_data.append(Data_CO)

    print('No of days = ' + str(no_of_days))
    return output_dataset_list, Prediction_date


# Replace these paths with the appropriate file paths
csv_file_path = r"C:\Users\vaibh\OneDrive\Documents\UCLA_Courses\M202A - Embedded Systems\LLMs_ReadingaMaterial\FineTuning\Dataset\AirQualityUCI_mod.csv"
#csv_file_path = r"C:\Users\vaibh\OneDrive\Documents\UCLA_Courses\M202A - Embedded Systems\LLMs_ReadingaMaterial\FineTuning\input.csv"

# Call the function
raw_data, Prediction_date = csv_to_dataset(csv_file_path)
save_the_file(raw_data, 'Raw_data.txt')

# print(len(raw_data))
# print(len(Prediction_date)) 

output = Remove_200_data(raw_data)
save_the_file(output, 'Raw_data_wo_200.txt')


# Extract the first part (list1) and the last elements (list2)
Image_dataset = [sublist[:-12] for sublist in output]  # Take all elements except the last
Next_day_result_dataset = [sublist[-12:] for sublist in output]   # Take the last element from each sublist

# print(len(Image_dataset))
# print(len(Next_day_result_dataset))

#json file data
json_list = []

# count is the each 5 day set we have. $ days of image generation and next day to predict
count = 0
for data in Image_dataset:
    #Find the average of the 5th day dataset
    result_avg = round( sum(Next_day_result_dataset[count]) / len(Next_day_result_dataset[count]) , 2)
    if count%2 == 0:
        m_or_n = 'm'
        img_path = 'Dataset/Images/Data_' + str(count) + '_' + m_or_n + '.jpg'
        data_item = {
            "id":f"train_{count}",
            "image_path": f"{img_path}",
            "query": f"This is the Scalogram of C0 over 4 days day time data. Based on this, whats the next day approximate average of CO data?",
            "answers": f"{result_avg}"
        }
    else:
        m_or_n = 'n'
        img_path = 'Dataset/Images/Data_' + str(count) + '_' + m_or_n + '.jpg'
        data_item = {
            "id":f"train_{count}",
            "image_path": f"{img_path}",
            "query": f"This is the Scalogram of C0 over 4 days night time data. Based on this, whats the next day approximate average of CO data?",
            "answers": f"{result_avg}"
        }
    # str_img = Prediction_date[int(count/2)].replace("/", "_") + '_' + m_or_n
    ############# uncomment below line to save the transformed images ###################
    # Scalogram_conv_out(data, img_path)

    json_list.append(data_item)
    count = count+1


json_file_path = r"C:\Users\vaibh\OneDrive\Documents\UCLA_Courses\M202A - Embedded Systems\LLMs_ReadingaMaterial\FineTuning\Output_dataset_img.json"


# Write to JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(json_list, json_file, indent=4)
