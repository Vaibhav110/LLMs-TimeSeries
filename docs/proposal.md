# Project Proposal

## 1. Motivation & Objective

Large Language Models (LLMs) are traditionally known for their prowess in handling text data, but their use in time series data analysis has yet to be explored much. Since these models excel at capturing sequential dependencies in text, they may also effectively model the sequential dependencies inherent in time series data. Time series data shares structural similarities with text, involving sequential patterns (e.g., trends, seasonality).
Hence, real-time, localised processing for time series tasks on edge devices (e.g., IoT sensors, wearables, and industrial automation) could greatly benefit from LLM deployment. Taking that forward, we plan to enable LLM for time series forecasting on embedded systems. We chose the Llama 3.2 1B [4] LLM, which is a pre-trained model that can run on edge devices via inputs in the form of raw data as well as images.


## 2. State of the Art & Its Limitations

Large Language Models (LLMs) are currently at the cutting edge of artificial intelligence research. Models such as GPT have demonstrated impressive capabilities in generating coherent and contextually relevant text based on the prompts they receive. However, direct connections between language modelling and time series forecasting remain undefined.
This gap stems from challenges like the lack of semantic context in time-series data, computational limitations, and LLMs’ difficulty processing numerical inputs. Also, many time-series LLMs convert the time-series data into text tokens, which could cause a loss in their ability to recognise temporal patterns.


## 3. Novelty & Rationale

These models leverage vast amounts of data and sophisticated architectures to understand context capture nuances, making them powerful tools across various applications. There are few models specifically catered for embedded devices. We plan to leverage those models, test them on time series and image-transformed data, and evaluate the results. 

Some works have been relevant to LLM-based time series models and smaller and more efficient neural models for time series. Based on that, our novelty is based on three broad aspects:
- Broadening LLM Application Scope by extending it for general time series analysis
- Expanding the time series tasks for both direct data or via Images transformed data
- Evaluation for an Edge device like a Phone


## 4. Potential Impact

The successful development of an improved LLM can revolutionise numerous fields. The ability to transform time series data can open doors for LLM Applications Beyond NLP expansion. It can also improve upon the capabilities of the edge devices. Hence, new standards for time analysis models have been set up.

It can have multiple real-life applications, also like:
- IoT and Smart Devices: Real-time anomaly detection
- Healthcare Monitoring: Wearable devices for vital signs tracking
- Finance and Retail: Demand forecasting



## 5. Challenges

What are the challenges and risks?
Conducting this experiment on LLMs for time series analysis on edge devices presents several challenges. The first hurdle for a beginner in LLMs is understanding how a Large Language Model operates, including the fine-tuning process and the necessary steps to get the model running effectively. Based on this foundation, there are other challenges like:
- Computational constrained environment like a phone
- Correct transformation for the Time series data with LLM compatibility
- Deployment challenges
- Model fine-tuning challenges like computation power

## 6. Requirements for Success

Any time series data should be okay, but we shall go forward with a sensor application.
We will use a pre-trained LLM model i.e. Llama 3.2 1B, which is a low-cost (less memory intensive) open-source fully trained model.
The hardware requirements are decent, such as having enough memory and 8GB RAM support and a GPU for fine-tuning the data.

The skills required are the following:
- Familiarity with embedded systems development, especially like phones for deep learning models.
- Proficiency with a variety of programming languages, including C/C++, Python. Applications like Matlab and Android Studio, and environments like Linux and OS.
- Understanding of the Deep learning models, mainly the LLMs, and the fine-tuning and tokenisation aspects. 


## 7. Metrics of Success

Success metric for LLMs on time series tasks across these configurations should be measured for both model performance and suitability for deployment on edge devices.
Key metrics for evaluation are:
- **Accuracy:** Deviation of predictions from actual values judged by the RMSE (root mean square value), etc
- **Latency:** Measured on the time required to generate the prediction critical for edge devices
- **Robustness of Fine-tuning:** Performance between fine-tuned and non-fine-tuned models to see if that is acceptable.
- **Affect based on Input data:** Track the model accuracy based on the RAW data vs image transformed data.

Overall, these performance parameters might be highly effective in evaluating all the configuration and LLMs models in general.


## 8. Execution Plan

There is a four-fold process to execute the whole project, mainly:

- Step 1: Data Processing
  - Prepare two datasets: raw time series data and image-transformed time series data.
  - Image transformation can be Spectrogram or Scalogram

- Step 2: Model Configuration
  - Case 1: Fine-tuned LLM on raw data.
  - Case 2: Non-fine-tuned LLM on raw data.
  - Case 3: Fine-tuned LLM on image-transformed data.
  - Case 4: Non-fine-tuned LLM on image-transformed data

- Step 3: Model Training & Testing
  - Train and evaluate each configuration to understand the performance
  - Running the model on an edge device constraints the environment
- Step 4: Evaluation Metrics
  - Evaluate the model based on the Accuracy and memory footprint



## 9. Related Work

### 9.a. Papers

There had been some work on the LLMs for time series data but that still needs to be defined. Here are a few papers which talk more:
- Are Language Models Actually Useful for Time Series Forecasting? [1]
  - As the name suggests, the paper evaluates if a time series is a fitting dataset for an LLM model. They concluded the paper that despite their significant computational cost, pre-trained LLMs do no better than models trained from scratch and do not represent the sequential dependencies in time series.
- SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition[2]
  -  Paper tried to address some key issues in Time series LLMs by introducing Sensor LLM, a two-stage framework to unlock LLMs’ potential for sensor data tasks. They enable SensorLLM to capture numerical changes, channel specific information, and sensor data of varying lengths.
- Towards Time-Series Reasoning with LLMs [3]
  - The paper aims to show that their model learns a latent representation that reflects specific time-series features (e.g. slope, frequency), as well as outperforming GPT-4o on a set of zero-shot reasoning tasks on a variety of domains

### 9.b. Datasets

For now, I decided to take the dataset present on the Kaggle: **Temperature Readings: IOT Devices** licensed under GNU Lesser General Public License. [6]

### 9.c. Software

- ExecuTorch solution by Pytorch [5]
   -It's an end-to-end solution that enables on-device inference capabilities across mobile and edge devices. They 
- Large Language Model: Llama 3.2 1B from Meta [4]
  - Llama 3.2 1B, a low-cost (less memory intensive) open-source fully trained model.
  The model was released in September, and it is still in the early stages of research by developers.
- Software/Libraries Used like Android Studio, python, Matlab, and GPU for fine-tuning


## 10. References

[1] Mingtian Tan, Mike A. Merrill, Vinayak Gupta. (2024). Are Language Models Actually Useful for Time Series Forecasting?

[2] Zechen Li, Shohreh Deldari, Linyao Chen, Hao Xue, Flora D. Salim. SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition. Proceedings of the 37th AAAI Conference on Artificial Intelligence (AAAI 2024).

[3] Winnie Chow, Lauren Gardiner, Haraldur T. Hallgrímsson. Towards Time Series Reasoning with LLMs (2024)

[4] Llama 3.2 Meta (2024) https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/

[5] Executorch by pytorch. https://github.com/pytorch/executorch

[6] Kaggle Temperature Readings: IOT Devices dataset https://www.kaggle.com/datasets/atulanandjha/temperature-readings-iot-devices/data


