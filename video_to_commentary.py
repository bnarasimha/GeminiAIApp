import io
import os
import cv2
from gtts import gTTS
from google.cloud import vision
from PIL import Image
import torch
from openai import OpenAI
from transformers import BlipProcessor, BlipForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
objects_extraction_dir = '.'

def analyze_frame(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to(device)

    generation_args = {
    "max_length": 500,  # Maximum length of the caption
    "num_beams": 5,    # Beam search with 5 beams
    "temperature": 1.0, # Sampling temperature
    "top_k": 50,       # Top-k sampling
    "top_p": 0.95,     # Top-p (nucleus) sampling
    "no_repeat_ngram_size": 2  # Prevent repetition of 2-grams
    }

    out = model.generate(**inputs, **generation_args)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

video_path = "/Users/nbadrinath/Documents/GeneralGithub/GeminiAIApp/mixkit-close-shot-of-a-soccer-player-shooting-a-penalty-43494-hd-ready.mp4"
vidcap = cv2.VideoCapture(video_path)
success, image = vidcap.read()
count = 0
frame_interval = 30  # Extract one frame every second if the video is 30fps
while success:
    if count % frame_interval == 0:
        cv2.imwrite(f"frame{count}.jpg", image)  # Save frame as JPEG file
    success, image = vidcap.read()
    count += 1

frame_descriptions = []
for i in range(0, count, frame_interval):
    description = analyze_frame(f'frame{i}.jpg')
    frame_descriptions.append(description)


import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

commentary = []
for description in frame_descriptions:
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=100,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a sports commentator, skilled in football"},
            {"role": "user", "content": f"Generate an insightful commentary based on this scene. Keep it short and exactly whats happening in the scene : {description}"}
        ]
    )
    commentary.append(completion.choices[0].message.content)


for i, comment in enumerate(commentary):
    tts = gTTS(comment, slow=False)
    tts.save(f'commentary_{i}.mp3')


# from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip

# video = VideoFileClip(video_path)
# commentary_audio_clips = [AudioFileClip(f'commentary_{i}.mp3') for i in range(len(commentary))]
# final_audio = concatenate_videoclips(commentary_audio_clips)
# final_video = video.set_audio(final_audio)
# final_video.write_videofile("final_video_with_commentary.mp4", codec="libx264")



















# client = vision.ImageAnnotatorClient()

# def analyze_frame(frame_path):
#     with io.open(frame_path, 'rb') as image_file:
#         content = image_file.read()
#     image = vision.Image(content=content)
#     response = client.label_detection(image=image)
#     labels = response.label_annotations
#     return ', '.join([label.description for label in labels])





