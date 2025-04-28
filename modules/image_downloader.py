from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                           QLabel, QFileDialog, QProgressBar, QListWidget,
                           QHBoxLayout, QSplitter, QFrame, QTextEdit,
                           QLineEdit, QGridLayout, QCheckBox, QGroupBox,
                           QRadioButton, QButtonGroup)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

def create_widget(parent=None):
    return ImageDownloaderWidget(parent)

class ImagePreview(QFrame):
    """Modern image preview panel"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        self.setObjectName("previewFrame")
        self.setMinimumWidth(200)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Preview title
        preview_label = QLabel("Image Preview")
        preview_label.setProperty("subtitle", "true")
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(preview_label)
        
        # Image preview area
        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(200)
        self.image_label.setStyleSheet("""
            border: 1px dashed palette(mid);
            border-radius: 4px;
            padding: 8px;
        """)
        layout.addWidget(self.image_label)
        
        # Image information
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
    
    def load_image(self, image_path):
        """Load and display an image - implementation details will be added later"""
        self.image_label.setText(f"Preview: {image_path}")
        self.info_label.setText("Image information will be displayed here")


class ImageDownloaderWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        
        title = QLabel("Image Downloader")
        title.setObjectName("page_title")
        layout.addWidget(title)
        
        description = QLabel("This module allows you to download images from URLs or web pages.")
        description.setObjectName("page_description")
        layout.addWidget(description)
        
        # Placeholder for now
        placeholder = QLabel("Image Downloader interface will be implemented here.")
        placeholder.setObjectName("placeholder_text")
        layout.addWidget(placeholder)
        
        layout.addStretch()
    
    def toggle_input_mode(self):
        """Toggle between URL list and webpage URL input modes"""
        if self.url_list_radio.isChecked():
            self.url_input_container.show()
            self.webpage_input_container.hide()
        else:
            self.url_input_container.hide()
            self.webpage_input_container.show()
    
    def toggle_rename_options(self, state):
        """Enable/disable rename options based on checkbox state"""
        self.prefix_input.setEnabled(state)
    
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder.setText(folder)
    
    def download_images(self):
        """Placeholder for download functionality"""
        # Just for UI demonstration
        self.status_label.setText("Simulating download...")
        self.progress.setValue(0)
        self.progress.setMaximum(100)
        
        # Display what would happen (actual implementation will come later)
        if self.url_list_radio.isChecked():
            urls = self.url_input.toPlainText().strip().split('\n')
            for i, url in enumerate(urls):
                if url.strip():
                    self.results_list.addItem(f"Would download: {url.strip()}")
            
            # Update UI to show what would happen
            self.status_label.setText(f"Download simulation: {len(urls)} images")
            self.progress.setValue(100)
        else:
            self.status_label.setText("Web page download simulation")
            self.results_list.addItem("Would scan: " + self.webpage_url.text())
            self.progress.setValue(100)
    
    def clear_results(self):
        """Clear the results list and reset UI"""
        self.results_list.clear()
        self.progress.setValue(0)
        self.status_label.setText("Ready")
        self.preview.image_label.setText("No image selected")
        self.preview.info_label.setText("")
