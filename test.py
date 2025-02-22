import base64
import matplotlib.pyplot as plt

from langchain.tools import BaseTool

import pyscreenshot as ImageGrab


def run(self):

    screenshot = ImageGrab.grab()

    screenshot_bytes = screenshot.convert('RGB').tobytes()

    base64_screenshot = base64.b64encode(screenshot_bytes).decode('utf-8')

    return f"Base64 image: {base64_screenshot}"



def main():
    screenshot = ImageGrab.grab()
    screenshot.show()

if __name__ == '__main__':
    main()
