import gradio as gr
import pandas as pd
import numpy as np
import os
from PIL import Image
import requests
from io import BytesIO
from imageai.Prediction import ImagePrediction
import csv

# Load the protected-birds.csv file
protected_birds = pd.read_csv('protected-birds.csv')
protected_birds = protected_birds.dropna()
protected_birds = protected_birds['Common Name'].str.lower().values

# Load the image ai model
model = ImagePrediction()
model.setModelTypeAsResNet()
model.setModelPath("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
model.loadModel()

# Function to check if the feather is from a protected bird
def check_feather(image):
    # Load the image
    response = requests.get(image)
    img = Image.open(BytesIO(response.content))
    img.save('image.jpg')
    
    # Make a prediction
    model_predictions, probabilities = model.predictImage('image.jpg', result_count=5)
    
    # Check if the feather is from a protected bird
    protected = False
    protected_birds_list = []
    for bird in protected_birds:
        for prediction in model_predictions:
            if bird in prediction.lower():
                protected = True
                protected_birds_list.append([bird, prediction, probabilities[model_predictions.index(prediction)]])
    
    if protected:
        # Display the results
        gr.Interface(fn=lambda: protected_birds_list, inputs="text", outputs="text").launch()
        user_verification = gr.inputs.Checkbox(label="Verify the results")
        if user_verification:
            # Output the result in a table format
            gr.Interface(fn=lambda: protected_birds_list, inputs=user_verification, outputs="html").launch()
            # Prompt user to make a report
            gr.Interface(fn=lambda: "If confident, report to: https://www.fws.gov/contact-us", inputs=user_verification, outputs="text").launch()
    else:
        gr.Interface(fn=lambda: "No protected bird detected", inputs=image, outputs="text").launch()

    return protected, protected_birds_list

# Gradio interface
image = gr.inputs.Image()
gr.Interface(fn=check_feather, inputs=image, outputs="text").launch()