# This file contains all shared functions and classes

import pandas as pd
import os
import subprocess
import re
import cv2
import pytesseract
import configparser
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
import pickle
import datetime
import sys
import random
import openpyxl
import math
import time
from hachoir.metadata import extractMetadata
from hachoir.parser import createParser
import logging
import xlwings as xw
import keyboard
from ultralytics import YOLO
import torch
import shutil

def ask_yes_no(text):
    global logger
    logger.debug(f'Running function ask_yes_no({text})')
    result: bool = messagebox.askyesno("Confirmation", text)
    return result

def create_dir(path):
    global logger
    logger.debug(f'Running function create_dir({path})')
    if not os.path.exists(path):
        os.makedirs(path)

def log_write():
    global logger
    # Create a logger instance
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a file handler that logs all messages, and set its formatter
    file_handler = logging.FileHandler('runtime.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create a console handler that logs only messages with level INFO or higher, and set its formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

log_write()