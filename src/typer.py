import pyautogui
import time
from pathlib import Path

project_folder = Path(__file__).parents[1]
with open(project_folder.joinpath('src/attacker.py')) as file:
    text = file.read()

# time.sleep(5)
# pyautogui.write(text, interval=0.1)
# text
