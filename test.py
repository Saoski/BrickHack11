import base64
import matplotlib.pyplot as plt

from langchain.tools import tool

import pyscreenshot as ImageGrab

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
