#!/usr/bin/env python3
import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QStackedWidget, QPushButton, QMessageBox
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QFontDatabase
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time

# Import UI modules
from ui.main_window import MainWindow
from modules.marketplace_x import MarketplaceXWidget
from modules.image_downloader import ImageDownloaderWidget
from modules.split_x import SplitXWidget

# Set up application-wide style and resources
def setup_application():
    # Create application instance
    app = QApplication(sys.argv)
    
    # Set application name and organization
    app.setApplicationName("BotX Pro")
    app.setOrganizationName("BotX Team")
    
    # Set app-wide stylesheet
    style_file = os.path.join(os.path.dirname(__file__), "ui", "styles", "dark_theme.qss")
    if os.path.exists(style_file):
        with open(style_file, "r") as f:
            app.setStyleSheet(f.read())
    
    # Create main window
    window = MainWindow()
    window.show()
    
    # Register modules
    window.register_module("marketplace_x", MarketplaceXWidget(window))
    window.register_module("image_downloader", ImageDownloaderWidget(window))
    window.register_module("split_x", SplitXWidget(window))
    
    # Execute application
    sys.exit(app.exec())

if __name__ == "__main__":
    setup_application()
