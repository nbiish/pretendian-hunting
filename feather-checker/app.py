import gradio as gr
import csv
import anthropic
import base64
import httpx

# Load the protected-birds.csv file into a list
protected_birds_list = []
with open('protected-birds.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # Skip the headers
    for row in reader:
        protected_birds_list.append(row[0].lower())  # Assuming 'Common Name' is in the first column

# Initialize the anthropic client
client = anthropic.Anthropic()

# Function to check if the feather is from a protected bird
def check_feather(image_url):
    # Load the image
    image_media_type = "image/jpeg"
    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
    
    # Make a prediction
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe this image."
                    }
                ],
            }
        ],
    )
    model_predictions = message.choices[0].message['content']['text']
    
    # Check if the feather is from a protected bird
    protected = False
    for bird in protected_birds_list:
        if bird in model_predictions.lower():
            protected = True
            protected_birds_list.append([bird, model_predictions])
    
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
gr.Interface(fn=check_feather, inputs=image_url, outputs="text").launch()