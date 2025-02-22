import base64
import datetime
import getpass
import os
import uuid

import pyscreenshot as ImageGrab
from langchain.tools import tool
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

hint_prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful math tutor that will get hints any explain concepts in any problems you are asked about"),
    ("human", "I am struggling with understanding this problem:\n"
              "{problem_description}\n"
              "Can you give me advice?")
])

# @tool
def take_screenshot():
    """Take a screenshot and return a description of the content within a red circle on the screenshot"""
    screenshot = ImageGrab.grab()
    screenshot.show()

    screenshot_bytes = screenshot.convert('RGB').tobytes()

    base64_screenshot = base64.b64encode(screenshot_bytes).decode('utf-8')

    return base64_screenshot


def generate_summary(image_url, prompt):
    # Initialize the model
    model = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)

    # Create a message with the image
    message = model.invoke(
        [
        HumanMessage(
            content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ]
            )
        ]
    )
    return message.content

def main():
    # Setup the API Key for OpenAI
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Provide your OpenAI API Key")

    # Open LangSmith
    uid = uuid.uuid4().hex[:6]
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    project_name = f"openai_image_summarization_{current_time}"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"]=project_name

    # Take a Screenshot
    screenshot = ImageGrab.grab()
    screenshot.show()

    # Get the image summary
    prompt = """What's in this image?"""
    res = generate_summary(screenshot, prompt)
    print(res)



if __name__ == '__main__':
    main()
