import base64
import datetime
import getpass
import os
import uuid
from mimetypes import guess_type

import pyscreenshot as ImageGrab
from langchain.tools import tool
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, \
    HumanMessagePromptTemplate

from dotenv import load_dotenv

import requests
import PIL
from langchain_core.prompts.image import ImagePromptTemplate

import cv2

load_dotenv()
key = os.getenv("OPENAI_KEY")

hint_prompt_template = ChatPromptTemplate([("system",
                                            "You are a helpful math tutor that will get hints any explain concepts in any problems you are asked about"),
    ("human", "I am struggling with understanding this problem:\n"
              "{problem_description}\n"
              "Can you give me advice?")])


# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    # Default to png
    if mime_type is None:
        mime_type = 'image/png'

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode(
            'utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


# @tool
def take_screenshot():
    """Take a screenshot and return a description of the content within a red circle on the screenshot"""
    screenshot = ImageGrab.grab()
    return screenshot

    # screenshot_bytes = screenshot.convert('RGB').tobytes()

    # base64_screenshot = base64.b64encode(screenshot_bytes).decode('utf-8')

    # return base64_screenshot


def generate_summary(image_url, prompt):
    # Initialize the model
    model = ChatOpenAI(model="gpt-4o", max_tokens=1024)

    # Create a message with the image
    message = model.invoke([HumanMessage(
        content=[{"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}, }])])

    return message.content

def find_rectangles(image_path):
    cascade = cv2.CascadeClassifier('path_to_harrcascade.xml')

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColot(image, cv2.COLOR_BGR2GRAY)

    rectangles = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    return rectangles


def main():
    # Setup the API Key for OpenAI
    os.environ["OPENAI_API_KEY"] = str(key)

    # Open LangSmith
    uid = uuid.uuid4().hex[:6]
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    project_name = f"openai_image_summarization_{current_time}"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name

    # Take a Screenshot
    screenshot = take_screenshot()
    screenshot_file_path = r"screenshots/screenshot.png"
    screenshot.save(screenshot_file_path)
    image_data_url = local_image_to_data_url(screenshot_file_path)

    # Get the image summary
    prompt = """What's in this image?"""
    res = generate_summary(image_data_url, prompt)
    print(res)


    # Find red rectangles from the image
    rectangles = find_rectangles(image_data_url)

if __name__ == '__main__':
    main()
