import gradio as gr
import csv
import gemini
import base64
import httpx

# Load the protected-birds.txt file into a dictionary
protected_birds_dict = {}
with open('protected-birds.txt', 'r') as txtfile:
    for line in txtfile:
        bird, confidence = line.strip().split(',')
        protected_birds_dict[bird.lower()] = float(confidence)

# Initialize the gemini client
def init_client(api_key):
    return gemini.Client(api_key)

# Function to check if the feather is from a protected bird
def check_feather(image_url, api_key):
    # Load the image
    image_media_type = "image/jpeg"
    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
    
    # Make a prediction
    client = init_client(api_key)
    response = client.predict(
        model_id="text-bison-001",
        inputs={
            "image": image_data,
            "question": "Describe this image."
        }
    )
    model_predictions = response["candidates"][0]["output"]
    
    # Check if the feather is from a protected bird
    protected = False
    protected_birds_list = []
    for bird, confidence in protected_birds_dict.items():
        if bird in model_predictions.lower():
            protected = True
            protected_birds_list.append([bird, confidence, model_predictions])
    
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
        gr.Interface(fn=lambda: "No protected bird detected", inputs=image_url, outputs="text").launch()

    return protected, protected_birds_list

# Gradio interface
image_url = gr.inputs.Textbox()
api_key = gr.inputs.Textbox(label="Google API Key")
gr.Interface(fn=check_feather, inputs=[image_url, api_key], outputs="text").launch()
