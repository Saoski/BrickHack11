import base64

from langchain.tools import tool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

import pyscreenshot as ImageGrab

hint_prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful math tutor that will get hints any explain concepts in any problems you are asked about"),
    ("human", "I am struggling with understanding this problem:\n"
              "{problem_description}\n"
              "Can you give me advice?")
])

@tool
def take_screenshot():
    """Take a screenshot and return a description of the content within a red circle on the screenshot"""
    screenshot = ImageGrab.grab()

    screenshot_bytes = screenshot.convert('RGB').tobytes()

    base64_screenshot = base64.b64encode(screenshot_bytes).decode('utf-8')

    return f"Base64 image: {base64_screenshot}"



def main():
    screenshot = ImageGrab.grab()
    screenshot.show()

if __name__ == '__main__':
    main()
