# Table of Contents
* Abstract
* [Introduction](#1-introduction)
* [Related Work](#2-related-work)
* [Technical Approach](#3-technical-approach)
* [Evaluation and Results](#4-evaluation-and-results)
* [Discussion and Conclusions](#5-discussion-and-conclusions)
* [References](#6-references)

# Abstract

LLMs, as Large language models, are recognized for their ability to generalize across a broad range of natural language tasks. Existing research has harnessed their extensive world knowledge and powerful pattern-recognition capabilities, often through the integration of specialized adapters and fine tuning for numerical input and projection layers for output generation. However, there remains limited exploration into whether LLMs can effectively handle more general time series tasks, either with or without task-specific fine-tuning applied to raw time series data or image-transformed representations. This study aims to assess LLM performance on edge devices by evaluating four different configurations: fine-tuning on raw data, no fine-tuning on raw data, fine-tuning on image-transformed data, and no fine-tuning on image-transformed data.
# 1. Introduction

### 1.1 Motvation and Objective
Large Language Models (LLMs) are traditionally known for their prowess in handling text data, but their use in time series data analysis has yet to be explored much. Since these models excel at capturing sequential dependencies in text, they may also effectively model the sequential dependencies inherent in time series data. Time series data shares structural similarities with text, involving sequential patterns (e.g., trends, seasonality). Hence, real-time, localised processing for time series tasks on edge devices (e.g., IoT sensors, wearables, and industrial automation) could greatly benefit from LLM deployment. Taking that forward, we plan to enable LLM for time series forecasting on embedded systems. We will use the existing models and fine tune them based on the time series data and understand the result. We chose the Llama 3.2 1B [4] LLM for text models and Llava 1.6 7B [8] for multimodal inputs. Both of these are small pre-trained model that can run on edge devices.

### 1.2 State of the Art & Its Limitations: 
Large Language Models (LLMs) are currently at the cutting edge of artificial intelligence research. Models such as GPT have demonstrated impressive capabilities in generating coherent and contextually relevant text based on the prompts they receive. However, direct connections between language modelling and time series forecasting is still under research without much positive outcome. This gap stems from challenges like the lack of semantic context in time-series data, computational limitations, and LLMs’ difficulty processing numerical inputs. Also, many time-series LLMs convert the time-series data into text tokens, which could cause a loss in their ability to recognise temporal patterns.
More details about the other models are mentioned in the Related work section.

### 1.3 Novelty & Rationale: 
 We plan to leverage existing pre trained models, fine tune and later test them on time series and image-transformed data, and evaluate the results. Some works have been relevant to LLM-based time series models and smaller and more efficient neural models for time series. Based on that, our novelty is based on four broad aspects:

* Broadening LLM Application Scope by extending it for general time series analysis
* Expanding the time series tasks for both direct data or via Images transformed data
* Understand how the changing context data can affect the prediction of the time series
* Test these on smaller models such that they can run on edge devices


### 1.4 Potential Impact: 
The successful development of an improved LLM can revolutionise numerous fields. The ability to transform time series data can open doors for LLM Applications Beyond NLP expansion. It can also improve upon the capabilities of the edge devices. Hence, new standards for time analysis models have been set up. It can have multiple real-life applications, also like:

* IoT and Smart Devices: Real-time anomaly detection
* Healthcare Monitoring: Wearable devices for vital signs tracking
* Finance and Retail: Demand forecasting

### 1.5 Challenges: 
Conducting this experiment on LLMs for time series analysis on edge devices presents several challenges. The first hurdle for a beginner in LLMs is understanding how a Large Language Model operates, including the fine-tuning process and the necessary steps to get the model running effectively. Based on this foundation, there are other challenges like:

* Running the model on an edge device
* Correct transformation for the Time series data with LLM compatibility
* Deployment challenges
* Data quality and quantity handing to make it relevant for the model
* Model fine-tuning challenges like computation power


### 1.6 Requirements for Success: 
Based on the usecase, we will use the Llama 3.2 1B and Llava 1.6 7B model which are open source pretrained models. The skills required are the following:
* Familiarity with embedded systems, especially like phones for deep learning models.
* Proficiency with a variety of programming languages, including C/C++, Python. Applications like Matlab and Android Studio, and environments like Linux and OS.
* Understanding of the Deep learning models, mainly the LLMs, and the fine-tuning and tokenisation aspects.
* GPU for fine tuning the LLM Model - 4 Nvidia H100 GPU provided by Mani Sir

### 1.7 Metrics of Success: 
Success is to understand how well the model undersand the data, based on the dataset context size and how does adding images help the model to learn the time series pattern. These understandings will be evaluated based on testing the model in 2 different ways:
* Testing it on the same datasets on which it was fine tuned one
* Testing it on similar but not the same dataset for which it was fine tuned on.

# 2. Related Work
There had been some work on the LLMs for time series data but that still needs to be defined. Here are a few papers which talk more:
- Are Language Models Actually Useful for Time Series Forecasting? [1]
  - As the name suggests, the paper evaluates if a time series is a fitting dataset for an LLM model. They concluded the paper that despite their significant computational cost, pre-trained LLMs do no better than models trained from scratch and do not represent the sequential dependencies in time series.
- SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition[2]
  -  Paper tried to address some key issues in Time series LLMs by introducing Sensor LLM, a two-stage framework to unlock LLMs’ potential for sensor data tasks. They enable SensorLLM to capture numerical changes, channel specific information, and sensor data of varying lengths.
- Towards Time-Series Reasoning with LLMs [3]
  - The paper aims to show that their model learns a latent representation that reflects specific time-series features (e.g. slope, frequency), as well as outperforming GPT-4o on a set of zero-shot reasoning tasks on a variety of domains

# 3. Technical Approach

The project is divided into 2 different approaches. Both of these approach work on 2 different datasets and can give different insight on how the LLMs understand the time series trends. Strategy I is the basic strategy which was initiated as the start of the project and to understand how the model behaves on the same dataset. Taking the inspiration we came up with Strategy II which will help us understand how well the model can perform for any similar diverse Air Quality dataset. Each approach will go through the same deployment method that is explained below. 

### 3.1 Deployment
![alt text](https://github.com/Vaibhav110/LLMs-TimeSeries/blob/main/docs/media/LoadingModel.png?raw=true)

Based on the dataflow above, out whole project is divided into 3 different parts.
- **Dataset Handling**

This step involves dataset cleaning, formatting the dataset that is readable by the LLM for fine tuning. Hence this step involves data processing to convert the data from csv file to a unique json format seperately for each training and testing. Later this dataset is pushed into Hugging face for better access.
- **Training Phase**

In this phase, we fine tune the model for the specific use case like time series datasets here. Based on the LLM model selected (Llama 3.2 and Llava 1.6 in our case), we load the training data into batches and feed it to the model. For our experiment, we choose 'pyTorch' as the framework and adoped the 'LoRA' as the fine tuning method. Once fine tuned, the model gives out the model pte file and the tokenizer file. These files can be used to run the model on phone.
 - **Testing Phase**

Once the model is fine tuned, we can test the model based on the testing dataset we created based on both the strategy. The model output is thencompared with the ground truth to calculate the mean absolute error and root mean square error.

### 3.2 Time Series to Image Conversion
There are 3 different time series to image conversion that we are going to use and evaluate how well the model performs as compared with pure text models.
- **Spectrogram**: Its a time-frequency representation, visualizing how the frequency content of a signal changes over time

![alt text](https://github.com/Vaibhav110/LLMs-TimeSeries/blob/main/docs/media/Image_Spectrogram.jpg?raw=true)
- **Scalogram**: It's a visualizing how the signal's energy content is distributed across different time scales.

![alt text](https://github.com/Vaibhav110/LLMs-TimeSeries/blob/main/docs/media/Image_Scalogram.jpg?raw=true)
- **MTF - Markov Transition Fields**:  Time series as a sequence of states, where the probability of transitioning to a new state wrt the current state.

![alt text](https://github.com/Vaibhav110/LLMs-TimeSeries/blob/main/docs/media/Image_MTF.jpg?raw=true)

### 3.3 Datasets
We are going to use multiple dataset to analyze all the different resutls.
- **Dataset_v1 (Strategy 1)**
  - Air Quality - UCI Machine Learning Repository [6]
  - This dataset 9358 instances of hourly averaged over 1 year responses of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device.
  - We divide this dataset divided into 2 different halves: Training (80%) and Testing (20%). Both text and multimodal will be fine tuned via the training dataset and later tested using the testing dataset

- **Dataset_v2 (Strategy 2)**
  - Daily air quality data from the US Environmental Protection Agency [7]
  - It contains daily data of various pollutant for multiple cities across multiple years
  - For this, we capture 3 years worth of data for a particular pollutant and a particular city and use it to fine tune the model. Later we will test the fine tuned model using the data from some other city for some other year altogether.

#### 3.3.1 Dataset_v1 Processing and Prompt types
This dataset contains an hourly values for 1 year from an air quality chemical multisensory Device. It is deployed in a significantly polluted area, at road level, within an Italian city. We took the NO2 data as it had the least amount of missing values. All the missing values were calculated based on linear interpolation method.

For each training datapoint, we took 4 days values as the prompt query and get the fifth day as the reponse. We divided each into night and day and seperated them out as different instances. Based on these we had multiple prompts examples to understand how the model perform based on different context.

- **[Text Based model]** Prediction of the next day feature, based on the 4 days average values and context about the data collection - Llama 3.2 1B
> {
>         "user": "Mentioned data is from Air Quality Chemical Multisensory Device which uses tungsten oxide as its sensing material and is designed to detect nitrogen dioxide (NO2) gas. It is deployed in a significantly polluted area, at road level, within an Italian city.  Average value on those four days is 1254, 1134, 1131 and 1317. Estimate the expected average Nitrogen Dioxide value for the subsequent day.",
>         "assistant": "1325"
> },
 - **[Text Based model]** Prediction of the next day feature value by providing the 4 day hourly data and context about the data collection - Llama 3.2 1B
>    {
>        "user": "Mentioned data is from Air Quality Chemical Multisensory Device which uses tungsten oxide as its sensing material and is designed to detect nitrogen dioxide (NO2) gas. It is deployed in a significantly polluted area, at road level, within an Italian city.  Average value on those four days is [1558.8, ... , 1298.1], [1521.1, ... , 1942.1], [1780.2, ... , 1845.2] and [1673.0,... , 1925.0]. Estimate the expected average Nitrogen Dioxide value for the subsequent day.",
 >       "assistant": "1784"
 >   },
  - **[Text Based model]** Same like the last prompt but will be running on the multimodal model - LLava 1.6 7B (To Compare how images affect the model given the same dataset and the same LLM)

 - **[Multimodal model]** Prediction for next day, based on 4 day average values and image transformed data. - Llava 1.6 7B
>    {
>        "id": "id_3",
>        "image_path_spectrogram": "Dataset/Images_Spectrogram/Data_3_n.jpg",
>        "query_spectrogram": "Analyze the provided Spectrogram of Nitrogen Dioxide night time data for four days. Average value on those four nights is 1624, 1730, 1632 and 1601. Estimate the expected average Nitrogen Dioxide value for the subsequent night.",
>       "answers": "1590"
>    },
 -  **[Text Based model]** Prediction for next day, based on 4 day average, image transformed data and more context about the dataset - Llava 1.6 7B
>    {
>        "id": "id_3",
>        "image_path_spectrogram": "Dataset/Images_Spectrogram/Data_3_n.jpg",
>        "query_spectrogram": "Mentioned data is from Air Quality Chemical Multisensory Device which uses tungsten oxide as its sensing material and is designed to detect nitrogen dioxide (NO2) gas. It is deployed in a significantly polluted area, at road level, within an Italian city. Analyze the provided Spectrogram of Nitrogen Dioxide night time data for four days. Average value on those four nights is 1624, 1730, 1632 and 1601. Estimate the expected average Nitrogen Dioxide value for the subsequent night.",
>       "answers": "1590"
>    },

#### 3.3.2 Dataset_v2 Processing and Prompt types

This analysis is a bit different then the previous type of testing. Its an AirQuality dataset from the  US Environmental Protection Agency which provides yearly everyday data for single pollutant for a city. We took the NO2 pollutant as it had a good trend of data. The values were in floating values. As undertstood by one of the literature review [3], LLMs find it tough to understand decimal values, hence we multipled all the values by 10 and converted them to interger numbers. We also took a far greater number of days, 10 in our case, as the prompt to detect the next day value.

We fine tuned our model using three years worth of data from Los Angeles city from 2016 to 2018. We created the testing dataset from the same source but used a different city San Fransciso because it has similar weather conditions like LA and a different year, 2023 for instance. We also tried multiple prompts for testing the fine tuned model to understand how well it has understood the trend. The training and testing prompts are mentioned below. I have only mentioned multimodal prompts, text based prompts are also similar to this
- **[Training Prompt]** Provided 10 days data points as part of the prompt along with details about those days like the days and the season.
>    {
>        "id": "id_3",
>        "image_path_spectrogram": "Dataset/Images_Spectrogram/Data_3.jpg",
>        "query_spectrogram": " Average value on Nitrogen Dioxide in Los Angeles on ten consecutive days Wednesday, Thursday, Saturday, Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday during the Winter season is 179, ... , 507.Analyze the provided Spectrogram of Nitrogen Dioxide for ten days and estimate the expected Nitrogen Dioxide value for the subsequent day that is Sunday.",
>       "answers": "411"
>    },
- **[Testing Prompt 1]** Provide 10 days data without the additional context about the data. This is to understand if the model still require all the context about the data to predict the next day value.
>    {
>        "id": "id_1",
>        "image_path_spectrogram": "Dataset/Images_Spectrogram/Data_1.jpg",
>        "query_spectrogram": "Average value on Nitrogen Dioxide on ten consecutive days is 394, .. , 414. Analyze the provided Spectrogram of Nitrogen Dioxide for ten days and estimate the expected average Nitrogen Dioxide value for the subsequent day.",
>        "answers": "355"
>    },
- **[Testing Prompt 2]** This prompt is similar to the training dataset, just the city and the year is differnt.
>    {
>        "id": "id_1",
>        "image_path_spectrogram": "Dataset/Images_Spectrogram/Data_1.jpg",
>        "query_spectrogram": " Average value on Nitrogen Dioxide in San Francisco on ten consecutive days Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, Monday, Tuesday, Wednesday, Thursday during the Winter season is 394, .. , 414.Analyze the provided Spectrogram of Nitrogen Dioxide for ten days and estimate the expected Nitrogen Dioxide value for the subsequent day that is Friday.",
>        "answers": "355"
>    },
- **[Testing Prompt 3]** Here instead of providing a total of 10 days, we give 6 days of data along with all other context about the data. Main aim is understand how well the model has understood the trend in the data and how well can it relate with the context
>    {
>        "id": "id_0",
>        "image_path_spectrogram": "Dataset/Images_Spectrogram/Data_0.jpg",
>        "query_spectrogram": " Average value on Nitrogen Dioxide in San Francisco on six consecutive days Friday, Saturday, Sunday, Monday, Tuesday, Wednesday during the Winter season is 145, ... , 238.Analyze the provided Spectrogram of Nitrogen Dioxide for six days and estimate the expected Nitrogen Dioxide value for the subsequent day that is Thursday.",
>       "answers": "249"
>    },

# 4. Evaluation and Results
Based on the two different strategy and the 2 different models, one based on pure text prompts and another one being multimodal, we got some interesting results. Here is the summary of them:


# 5. Discussion and Conclusions

# 6. References
[1] Mingtian Tan, Mike A. Merrill, Vinayak Gupta. (2024). Are Language Models Actually Useful for Time Series Forecasting?

[2] Zechen Li, Shohreh Deldari, Linyao Chen, Hao Xue, Flora D. Salim. SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition. Proceedings of the 37th AAAI Conference on Artificial Intelligence (AAAI 2024).

[3] Winnie Chow, Lauren Gardiner, Haraldur T. Hallgrímsson. Towards Time Series Reasoning with LLMs (2024)

[4] Llama 3.2 Meta (2024) https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/

[5] Executorch by pytorch. https://github.com/pytorch/executorch

[6] Air Quality - UCI Machine Learning Repository https://archive.ics.uci.edu/dataset/360/air+quality

[7] US Environmental Protection Agency
https://www.epa.gov/outdoor-air-quality-data/download-daily-data

[8] LLava 1.6 7B 
https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf


