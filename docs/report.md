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
Large Language Models (LLMs) are traditionally known for their prowess in handling text data, but their use in time series data analysis has yet to be explored much. Since these models excel at capturing sequential dependencies in text, they may also effectively model the sequential dependencies inherent in time series data. Time series data shares structural similarities with text, involving sequential patterns (e.g., trends, seasonality). Hence, real-time, localised processing for time series tasks on edge devices (e.g., IoT sensors, wearables, and industrial automation) could greatly benefit from LLM deployment. Taking that forward, we plan to enable LLM for time series forecasting on embedded systems. We will use the existing models and fine tune them based on the time series data and understand the result. We chose the Llama 3.2 1B [4] LLM for text models and Llava 1.6 7B [7] for multimodal inputs. Both of these are small pre-trained model that can run on edge devices.

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

# 4. Evaluation and Results

# 5. Discussion and Conclusions

# 6. References
[1] Mingtian Tan, Mike A. Merrill, Vinayak Gupta. (2024). Are Language Models Actually Useful for Time Series Forecasting?

[2] Zechen Li, Shohreh Deldari, Linyao Chen, Hao Xue, Flora D. Salim. SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition. Proceedings of the 37th AAAI Conference on Artificial Intelligence (AAAI 2024).

[3] Winnie Chow, Lauren Gardiner, Haraldur T. Hallgrímsson. Towards Time Series Reasoning with LLMs (2024)

[4] Llama 3.2 Meta (2024) https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/

[5] Executorch by pytorch. https://github.com/pytorch/executorch

[6] Kaggle Temperature Readings: IOT Devices dataset https://www.kaggle.com/datasets/atulanandjha/temperature-readings-iot-devices/data.

[7] LLava 1.6 7B 
https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf


