# EMI-Sensor-Simulation-Augmentation-and-Inversion
This repository stores the code and data used to build an entire inversion pipeline using a Convolutional Neural Network Architecture

1. Performance Metrics.py: This code is was used to identify the performance metrics such as MAE or MSAE for the the difference between predicted and actual conductivity
2. CNN Inversion Model.py: This is the overarching program which takes a training data and test data input, before making a conductivity vs depth prediction output
3. Tikhonov Inversion.py: This program is another inversion model used as an industry comparison to the CNN model.
4. Temperature and Noise Application to Sim Data.py: This program takes forward modelled data produced by Marco 4.4.0 and applies temperature drift and noise effects to replicate the real tool behaviour of the downhole swept frequency tool
5. Dataset Expander.py: This program expands original forward modelled data from Marco 4.4.0 in to a larger dataset whilst maintaining shape in order to allow realisitc temp and noise fluctation augmentation similar to the downhole swept frequency tool.
6. Temp and Noise Behaviour Extraction from Lab Data: This program is used to identify temperature drift, instrumental gain, and noise scales across all frequencies based on lab data collected by the downhole swept frequency tool. These parameters are later used for data augmentation on synthetic forward modelled data (used in Temperature and Noise Application to Sim Data.py)
7. Training Data.csv: This is the downhole swept frequency lab experiment data set used for model training
8. SimXTestX+Tool.csv: These files are the synthetic borehole datasets with data augmentation applied to mimic real sensor behaviour.
   
