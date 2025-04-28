from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLabel, QStackedWidget, QFrame, 
                           QSizePolicy, QSpacerItem, QScrollArea, QGridLayout)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap

from ui.themes import apply_theme, get_system_theme
from utils.config import get_config

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.modules = {}
        self.current_module = None
        self.setup_ui()
        self.load_theme()
        
    def setup_ui(self):
        # Window setup
        self.setWindowTitle("BotX Pro")
        self.setMinimumSize(1400, 880)
        
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create main layout
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Create sidebar
        self.sidebar = self.create_sidebar()
        self.main_layout.addWidget(self.sidebar)
        
        # Create content area
        self.content_area = QStackedWidget()
        self.main_layout.addWidget(self.content_area)
        
        # Create welcome screen
        self.welcome_screen = self.create_welcome_screen()
        self.content_area.addWidget(self.welcome_screen)
        
        # Dictionary to hold module references
        self.modules = {}
        
    def create_sidebar(self):
        # Create sidebar container
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(250)
        
        # Create sidebar layout
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        
        # Add logo and title
        logo_container = QFrame()
        logo_container.setObjectName("logo_container")
        logo_layout = QVBoxLayout(logo_container)
        
        logo_label = QLabel("<b>BotX Pro</b>")
        logo_label.setObjectName("logo_label")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Make subtitle responsive
        subtitle_label = QLabel("Image Processing and Quality Check")
        subtitle_label.setObjectName("subtitle_label")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setWordWrap(True)  # Enable word wrapping
        subtitle_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)  # Allow horizontal expansion
        
        made_in_label = QLabel("Made in India")
        made_in_label.setObjectName("subtitle_label")
        made_in_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        logo_layout.addWidget(logo_label)
        logo_layout.addWidget(subtitle_label)
        logo_layout.addWidget(made_in_label)
        sidebar_layout.addWidget(logo_container)
        
        # Add navigation buttons
        nav_container = QFrame()
        nav_container.setObjectName("nav_container")
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(10, 20, 10, 20)
        nav_layout.setSpacing(5)
        
        # Home button
        home_button = QPushButton("Home")
        home_button.setObjectName("nav_button")
        home_button.setProperty("active", True)
        home_button.clicked.connect(lambda: self.show_module("home"))
        nav_layout.addWidget(home_button)
        
        # Image Downloader button
        downloader_button = QPushButton("Image Downloader")
        downloader_button.setObjectName("nav_button")
        downloader_button.clicked.connect(lambda: self.show_module("image_downloader"))
        nav_layout.addWidget(downloader_button)
        
        # MarketplaceX button
        marketplace_button = QPushButton("MarketplaceX")
        marketplace_button.setObjectName("nav_button")
        marketplace_button.clicked.connect(lambda: self.show_module("marketplace_x"))
        nav_layout.addWidget(marketplace_button)
        
        # SplitX button
        split_x_button = QPushButton("SplitX")
        split_x_button.setObjectName("nav_button")
        split_x_button.clicked.connect(lambda: self.show_module("split_x"))
        nav_layout.addWidget(split_x_button)
        
        # Add spacer to push the rest to the bottom
        nav_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        # Theme selector and version info
        footer_container = QFrame()
        footer_container.setObjectName("footer_container")
        footer_layout = QVBoxLayout(footer_container)
        
        version_label = QLabel("Version 1.0.0")
        version_label.setObjectName("version_label")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        footer_layout.addWidget(version_label)
        
        # Add containers to sidebar
        sidebar_layout.addWidget(nav_container)
        sidebar_layout.addWidget(footer_container)
        
        return sidebar
    
    def create_welcome_screen(self):
        """Create the welcome screen widget."""
        welcome_widget = QWidget()
        welcome_layout = QVBoxLayout(welcome_widget)
        welcome_layout.setContentsMargins(30, 30, 30, 30)
        welcome_layout.setSpacing(20)
        
        # Welcome header
        welcome_header = QLabel("Welcome to BotX Pro")
        welcome_header.setObjectName("welcome_header")
        welcome_layout.addWidget(welcome_header)
        
        welcome_subheader = QLabel("Select a module to get started:")
        welcome_subheader.setObjectName("welcome_subheader")
        welcome_layout.addWidget(welcome_subheader)
        
        # Create module cards container
        cards_container = QWidget()
        cards_layout = QGridLayout(cards_container)
        cards_layout.setSpacing(20)
        
        # Add module cards - SplitX first, then other modules
        # SplitX (first position)
        self.create_module_card(cards_layout, 0, 0, "SplitX", 
                              "Split and distribute images to multiple designers", 
                              "scissors", "split_x")
        
        # MarketplaceX (second position)
        self.create_module_card(cards_layout, 0, 1, "MarketplaceX", 
                              "Download content from marketplace", 
                              "cloud-download", "marketplace_x")
        
        # Image Downloader (third position)
        self.create_module_card(cards_layout, 1, 0, "Image Downloader", 
                              "Download image content from the web", 
                              "download", "image_downloader")
        
        # Empty space for future modules
        cards_layout.setColumnStretch(0, 1)
        cards_layout.setColumnStretch(1, 1)
        cards_layout.setRowStretch(2, 1)
        
        welcome_layout.addWidget(cards_container)
        welcome_layout.addStretch()
        
        # Footer with version and copyright
        footer = QLabel("Version 1.0.0 | © 2025 Developed by Creative Marketing and COE Team at Saks Global")
        footer.setObjectName("welcome_footer")
        footer.setAlignment(Qt.AlignmentFlag.AlignRight)
        welcome_layout.addWidget(footer)
        
        return welcome_widget
    
    def create_module_card(self, layout, row, col, title, description, icon_name, module_name):
        card = QFrame()
        card.setObjectName("feature_card")
        
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        
        title_label = QLabel(title)
        title_label.setObjectName("card_title")
        
        desc_label = QLabel(description)
        desc_label.setObjectName("card_description")
        desc_label.setWordWrap(True)
        
        open_button = QPushButton("Open Tool")
        open_button.setObjectName("primary_button")
        open_button.clicked.connect(lambda: self.show_module(module_name))
        
        card_layout.addWidget(title_label)
        card_layout.addWidget(desc_label)
        card_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        card_layout.addWidget(open_button)
        
        layout.addWidget(card, row, col)
        
    def register_module(self, name, widget):
        """Register a module to be shown in the content area"""
        self.modules[name] = widget
        self.content_area.addWidget(widget)
    
    def show_module(self, name):
        """Show a specific module in the content area"""
        # Update sidebar button states
        for button in self.sidebar.findChildren(QPushButton, "nav_button"):
            if name in button.text().lower().replace(" ", "_"):
                button.setProperty("active", True)
            else:
                button.setProperty("active", False)
            
            # Force style update
            button.style().unpolish(button)
            button.style().polish(button)
        
        # Show the appropriate widget
        if name == "home":
            self.content_area.setCurrentWidget(self.welcome_screen)
        elif name in self.modules:
            self.content_area.setCurrentWidget(self.modules[name])
    
    def load_theme(self):
        """Load the saved theme"""
        theme = self.config.get("theme", "Dark")  # Default to Dark theme
        self.apply_current_theme()
    
    def apply_current_theme(self):
        """Apply the current theme"""
        theme = self.config.get("theme", "Dark")  # Default to Dark theme
        apply_theme(self, theme)
