import base64
import datetime
import getpass
import os
import uuid
from mimetypes import guess_type

import numpy as np
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
from matplotlib import pyplot as plt

import cv2

load_dotenv()
key = os.getenv("OPENAI_KEY")

hint_prompt_template = ChatPromptTemplate([("system",
                                            "You are a helpful math tutor that will get hints any explain concepts in any problems you are asked about"),
                                           ("human",
                                            "I am struggling with understanding this problem:\n"
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


def generate_summary(image_url, prompt, model):
    # Create a message with the image
    message = model.invoke([HumanMessage(
        content=[{"type": "text", "text": prompt},
                 {"type": "image_url", "image_url": {"url": image_url}, }])])

    return message.content


def provide_hints(image_url, model):
    """Returns a list of possible hints for the given image"""
    message = model.invoke([HumanMessage(
        content=[{"type": "text",
                  "text": "Can you give me advice on how to solve this problem? Don't repeat the problem back. Just immediately respond with the first step and put each step on a new line."},
                 {"type": "image_url", "image_url": {"url": image_url}, }]
    )])
    print(message.content)
    return message.content.split("\n")


def test_image_prompts(model):
    data_directory = "data"
    for file in os.listdir(data_directory):
        full_path = os.path.join(data_directory, file)
        full_path = full_path.replace("\\", "/")
        if file.startswith("graph"):
            print("testing image:", file)
            image_data_url = local_image_to_data_url(full_path)
            prompt = "What is this graph"
            res = generate_summary(image_data_url, prompt, model)
            print(res)


def find_rectangles(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rectangles, threshold = cv2.threshold(gray_image, 50, 255, 0)

    return rectangles, threshold


def detect_red_rectangle(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 100, 100])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    largest_contour = 0, 0, 0, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            largest_contour = x, y, w, h

    return largest_contour


def test_red_rectangles(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x, y, w, h = detect_red_rectangle(image)
    cropped_image = image[y:y + h, x:x + w]
    # plt.imshow(cropped_image)
    # plt.show()
    return cropped_image


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
    # image_data_url = local_image_to_data_url(screenshot_file_path)

    # Initialize the model
    model = ChatOpenAI(model="gpt-4o", max_tokens=1024)

    # Get the image summary
    prompt = """What's in this image?"""
    # res = generate_summary(image_data_url, prompt, model)
    # print(res)

    # Find red rectangles from the image
    cropped_image = test_red_rectangles(screenshot_file_path)

    c_image = PIL.Image.fromarray(cropped_image)
    c_image.save("crop.png")

    data_url = local_image_to_data_url("crop.png")

    # res = generate_summary(data_url, prompt, model)
    hints = provide_hints(data_url, model)
    # print(res)
    print(hints)


if __name__ == '__main__':
    main()
