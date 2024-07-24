import PIL.Image
import gradio as gr
import base64
import time
import os
import google.generativeai as genai
import time

genai.configure(api_key = os.environ['GOOGLE_API_KEY'])

video_file_name = "mixkit-close-shot-of-a-soccer-player-shooting-a-penalty-43494-hd-ready.mp4"

print(f"Uploading file...")
video_file = genai.upload_file(path=video_file_name)
print(f"Completed upload: {video_file.uri}")



while video_file.state.name == "PROCESSING":
    print('.', end='')
    time.sleep(10)
    video_file = genai.get_file(video_file.name)

if video_file.state.name == "FAILED":
  raise ValueError(video_file.state.name)

file = genai.get_file(name=video_file.name)
print(f"Retrieved file '{file.display_name}' as: {video_file.uri}")

# Create the prompt.
prompt = "Describe this video."

# The Gemini 1.5 models are versatile and work with multimodal prompts
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# Make the LLM request.
print("Making LLM inference request...")
response = model.generate_content([video_file, prompt],
                                  request_options={"timeout": 600})
print(response.text)

