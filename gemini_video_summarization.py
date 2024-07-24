import vertexai
from vertexai.generative_models import GenerativeModel, Part

project_id = "geminiproject-427812"

vertexai.init(project=project_id, location="us-central1")

model = GenerativeModel(model_name="gemini-1.5-flash-001")

prompt = """
Provide a description of the video.
The description should also contain anything important which people say in the video.
"""

video_file_uri = "mixkit-close-shot-of-a-soccer-player-shooting-a-penalty-43494-hd-ready.mp4"
video_file = Part.from_uri(video_file_uri, mime_type="video/mp4")

contents = [video_file, prompt]

response = model.generate_content(contents)
print(response.text)