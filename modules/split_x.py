import os
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QLineEdit, QFileDialog, QProgressBar, 
                           QGroupBox, QScrollArea, QMessageBox, QFrame,
                           QTextEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QIntValidator, QIcon
from PIL import Image
import subprocess

class ImageProcessor(QThread):
    progress_updated = pyqtSignal(int, str, dict)
    processing_complete = pyqtSignal(dict)
    scan_progress = pyqtSignal(int)

    def __init__(self, source_folder, num_designers):
        super().__init__()
        self.source_folder = source_folder
        self.num_designers = num_designers
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
        self.stats = {
            'total_images': 0,
            'white_background': 0,
            'non_white_background': 0,
            'extensions': {},
            'designer_files': {}
        }

    def run(self):
        try:
            start_time = time.time()
            
            # Create designer folders
            for i in range(self.num_designers):
                folder_path = os.path.join(self.source_folder, f'Designer_{i+1}')
                os.makedirs(folder_path, exist_ok=True)
                self.stats['designer_files'][f'Designer_{i+1}'] = []

            # First scan to count files and group by file ID
            image_groups = {}  # Dictionary to store groups of images by file ID
            image_files = []
            for root, _, files in os.walk(self.source_folder):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in self.supported_formats:
                        # Extract file ID from filename
                        file_id = self.extract_file_id(file)
                        if file_id:
                            if file_id not in image_groups:
                                image_groups[file_id] = []
                            image_groups[file_id].append(os.path.join(root, file))
                            image_files.append(os.path.join(root, file))
                            self.stats['extensions'][ext] = self.stats['extensions'].get(ext, 0) + 1
                            self.scan_progress.emit(len(image_files))

            self.stats['total_images'] = len(image_files)
            processed = 0

            # Sort groups by ID to ensure consistent distribution
            sorted_groups = sorted(image_groups.items())
            total_groups = len(sorted_groups)
            
            # Calculate how many groups each designer should get
            groups_per_designer = total_groups // self.num_designers
            extra_groups = total_groups % self.num_designers

            # Distribute groups to designers
            current_designer = 0
            for i, (file_id, group_files) in enumerate(sorted_groups):
                # Determine which designer gets this group
                if i < (groups_per_designer + 1) * extra_groups:
                    current_designer = i // (groups_per_designer + 1)
                else:
                    current_designer = (i - extra_groups) // groups_per_designer

                designer_folder = os.path.join(self.source_folder, f'Designer_{current_designer + 1}')
                
                # Process all files in this group
                for image_path in group_files:
                    image_file = os.path.basename(image_path)
                    dest_path = os.path.join(designer_folder, image_file)
                    self.stats['designer_files'][f'Designer_{current_designer + 1}'].append(image_file)

                    try:
                        # Move file first
                        os.rename(image_path, dest_path)
                        
                        # Then apply tag based on background
                        if self.is_white_background(dest_path):
                            self.stats['white_background'] += 1
                            self.apply_mac_tag(dest_path, 6)  # Green tag for white background
                        else:
                            self.stats['non_white_background'] += 1
                            self.apply_mac_tag(dest_path, 4)  # Blue tag for non-white background
                        
                        processed += 1
                        elapsed_time = time.time() - start_time
                        self.progress_updated.emit(processed, self.format_time(elapsed_time), self.stats)

                    except Exception as e:
                        print(f"Error processing {image_file}: {str(e)}")

            # Create Excel report
            self.create_excel_report(start_time)
            self.processing_complete.emit(self.stats)

        except Exception as e:
            print(f"Error in processing thread: {str(e)}")

    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def apply_mac_tag(self, file_path, tag_index):
        try:
            # Convert the file path to a format that AppleScript can understand
            file_path = file_path.replace('"', '\\"')
            script = f"""osascript -e 'tell application "Finder" to set label index of (POSIX file "{file_path}" as alias) to {tag_index}'"""
            os.system(script)
        except Exception as e:
            print(f"Error applying tag: {str(e)}")

    def is_white_background(self, image_path):
        """
        Check if either top-left or top-right corner of the image has a white background.
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                pixels = img.load()
                
                # Check top-left corner (5x5 pixels)
                is_left_white = True
                for x in range(5):
                    for y in range(5):
                        if pixels[x, y][:3] != (255, 255, 255):  # Compare RGB only
                            is_left_white = False
                            break
                    if not is_left_white:
                        break
                
                # Check top-right corner (5x5 pixels)
                is_right_white = True
                for x in range(width-5, width):
                    for y in range(5):
                        if pixels[x, y][:3] != (255, 255, 255):  # Compare RGB only
                            is_right_white = False
                            break
                    if not is_right_white:
                        break
                
                return is_left_white or is_right_white
        except Exception as e:
            print(f"Error checking white background: {e}")
            return False

    def extract_file_id(self, filename):
        """
        Extract the file ID from the filename.
        Returns the first 13 characters if it's a digit, or the first 12 characters if it's alphanumeric.
        """
        # Check for 13-digit number
        if len(filename) >= 13 and filename[:13].isdigit():
            return filename[:13]
        # Check for 12-character alphanumeric
        elif len(filename) >= 12:
            return filename[:12]
        return None

    def create_excel_report(self, start_time):
        try:
            max_files = max(len(files) for files in self.stats['designer_files'].values()) if self.stats['designer_files'] else 0
            data = {designer: files + [''] * (max_files - len(files)) 
                   for designer, files in self.stats['designer_files'].items()}
            
            df = pd.DataFrame(data)
            excel_path = os.path.join(self.source_folder, 'SplitImg_Report.xlsx')
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Designer Files sheet
                df.to_excel(writer, sheet_name='Designer Files', index=False)
                
                # Format extensions count for display
                extensions_text = ', '.join(f"{ext} ({count})" for ext, count in self.stats['extensions'].items())
                
                # Calculate total processing time
                total_time = time.time() - start_time
                hours = int(total_time // 3600)
                minutes = int((total_time % 3600) // 60)
                seconds = int(total_time % 60)
                processing_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                # Create summary sheet
                summary_data = {
                    'Metric': [
                        'Total Images Processed',
                        'White Background Images',
                        'Non-White Background Images',
                        'Supported Extensions',
                        'Total Processing Time'
                    ],
                    'Value': [
                        self.stats['total_images'],
                        self.stats['white_background'],
                        self.stats['non_white_background'],
                        extensions_text,
                        processing_time
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Log successful report creation
            print(f"Excel report created: {excel_path}")
            return excel_path
            
        except Exception as e:
            print(f"Error creating Excel report: {str(e)}")
            return None


class SplitXWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize state variables
        self.processor = None
        
        # Set up the user interface
        self.setup_ui()
        
        # Add initial log entry with proper method
        self.add_log("SplitX module initialized")
    
    def setup_ui(self):
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Left panel - Settings and Stats
        settings_panel = QFrame()
        settings_panel.setObjectName("settings_panel")
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setSpacing(15)
        
        # Title
        title_label = QLabel("SplitX")
        title_label.setObjectName("page_title")
        settings_layout.addWidget(title_label)
        
        desc_label = QLabel("Split image files across multiple designer folders with automatic background detection")
        desc_label.setObjectName("page_description")
        desc_label.setWordWrap(True)
        settings_layout.addWidget(desc_label)
        
        settings_layout.addSpacing(10)
        
        # Scroll area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(20)
        
        # Designer configuration
        designer_group = QGroupBox("Designer Configuration")
        designer_group.setObjectName("settings_group")
        designer_layout = QVBoxLayout(designer_group)
        
        designer_label = QLabel("Number of Designers (1-60):")
        designer_label.setObjectName("setting_label")
        
        self.designer_input = QLineEdit()
        self.designer_input.setValidator(QIntValidator(1, 60))
        self.designer_input.setMaxLength(2)
        self.designer_input.setPlaceholderText("Enter number between 1-60")
        self.designer_input.textChanged.connect(self.validate_designer_input)
        
        designer_layout.addWidget(designer_label)
        designer_layout.addWidget(self.designer_input)
        
        scroll_layout.addWidget(designer_group)
        
        # Input folder section
        folder_group = QGroupBox("Source Folder")
        folder_group.setObjectName("settings_group")
        folder_layout = QVBoxLayout(folder_group)
        
        folder_desc = QLabel("Select the folder containing images to process:")
        folder_desc.setObjectName("setting_label")
        folder_layout.addWidget(folder_desc)
        
        folder_input_layout = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_input.setReadOnly(True)
        self.folder_input.setPlaceholderText("No folder selected")
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_folder)
        
        folder_input_layout.addWidget(self.folder_input)
        folder_input_layout.addWidget(browse_button)
        folder_layout.addLayout(folder_input_layout)
        
        scroll_layout.addWidget(folder_group)
        
        # Progress group
        progress_group = QGroupBox("Progress")
        progress_group.setObjectName("settings_group")
        progress_layout = QVBoxLayout(progress_group)
        
        self.scan_progress_label = QLabel("Scanning Images...")
        self.scan_progress_label.setObjectName("progress_label")
        
        self.scan_progress_bar = QProgressBar()
        self.scan_progress_bar.setObjectName("progress_bar")
        
        self.process_progress_label = QLabel("Processing Images...")
        self.process_progress_label.setObjectName("progress_label")
        
        self.process_progress_bar = QProgressBar()
        self.process_progress_bar.setObjectName("progress_bar")
        
        progress_layout.addWidget(self.scan_progress_label)
        progress_layout.addWidget(self.scan_progress_bar)
        progress_layout.addWidget(self.process_progress_label)
        progress_layout.addWidget(self.process_progress_bar)
        
        scroll_layout.addWidget(progress_group)
        
        # Statistics group
        stats_group = QGroupBox("Statistics")
        stats_group.setObjectName("settings_group")
        stats_layout = QVBoxLayout(stats_group)
        
        self.total_images_label = QLabel("Total Images Processed: 0")
        self.white_bg_label = QLabel("White Background Images: 0")
        self.non_white_bg_label = QLabel("Non-White Background Images: 0")
        self.time_label = QLabel("Time Taken: 00:00:00")
        self.extensions_label = QLabel("Supported Extensions: .png, .jpg, .jpeg, .bmp, .gif, .tiff")
        
        stats_layout.addWidget(self.total_images_label)
        stats_layout.addWidget(self.white_bg_label)
        stats_layout.addWidget(self.non_white_bg_label)
        stats_layout.addWidget(self.time_label)
        stats_layout.addWidget(self.extensions_label)
        
        scroll_layout.addWidget(stats_group)
        
        # Actions section
        actions_group = QGroupBox("Actions")
        actions_group.setObjectName("settings_group")
        actions_layout = QVBoxLayout(actions_group)
        
        self.run_button = QPushButton("Start Processing")
        self.run_button.setObjectName("primary_button")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.start_processing)
        
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_application)
        
        actions_layout.addWidget(self.run_button)
        actions_layout.addWidget(reset_button)
        
        scroll_layout.addWidget(actions_group)
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        settings_layout.addWidget(scroll)
        
        # Right panel - Logs
        log_panel = QFrame()
        log_panel.setObjectName("log_panel")
        log_layout = QVBoxLayout(log_panel)
        
        log_header = QLabel("Activity Log")
        log_header.setObjectName("panel_header")
        
        # Replace QScrollArea with QTextEdit for logs
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setObjectName("log_text")
        
        clear_button = QPushButton("Clear Log")
        clear_button.clicked.connect(self.clear_log)
        
        log_layout.addWidget(log_header)
        log_layout.addWidget(self.log_text)
        log_layout.addWidget(clear_button)
        
        # Add panels to main layout
        main_layout.addWidget(settings_panel, 3)
        main_layout.addWidget(log_panel, 2)
    
    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.folder_input.setText(folder)
            self.validate_inputs()
            self.add_log(f"Source folder selected: {folder}")
    
    def validate_designer_input(self):
        """Validate the designer input and update button state"""
        self.validate_inputs()
    
    def validate_inputs(self):
        """Validate all inputs and enable/disable run button accordingly"""
        designer_valid = False
        try:
            if self.designer_input.text():
                num_designers = int(self.designer_input.text())
                designer_valid = 1 <= num_designers <= 60
        except ValueError:
            designer_valid = False
        
        folder_valid = bool(self.folder_input.text())
        
        self.run_button.setEnabled(designer_valid and folder_valid)
    
    def start_processing(self):
        try:
            num_designers = int(self.designer_input.text())
            source_folder = self.folder_input.text()
            
            if 1 <= num_designers <= 60 and source_folder:
                self.processor = ImageProcessor(source_folder, num_designers)
                self.processor.progress_updated.connect(self.update_progress)
                self.processor.processing_complete.connect(self.processing_complete)
                self.processor.scan_progress.connect(self.update_scan_progress)
                self.processor.start()
                self.run_button.setEnabled(False)
                self.reset_progress()
                self.add_log(f"Starting processing with {num_designers} designers")
            else:
                self.add_log("Error: Please check your inputs")
        except ValueError:
            self.add_log("Error: Invalid number of designers")
        except Exception as e:
            self.add_log(f"Error: {str(e)}")
    
    def reset_progress(self):
        """Reset progress bars and statistics"""
        self.scan_progress_bar.setValue(0)
        self.process_progress_bar.setValue(0)
        self.total_images_label.setText("Total Images Processed: 0")
        self.white_bg_label.setText("White Background Images: 0")
        self.non_white_bg_label.setText("Non-White Background Images: 0")
        self.time_label.setText("Time Taken: 00:00:00")
    
    def update_scan_progress(self, count):
        """Update the scan progress bar"""
        self.scan_progress_bar.setValue(count)
        self.scan_progress_label.setText(f"Scanning Images... ({count} files found)")
        self.add_log(f"Found {count} image files")
    
    def update_progress(self, processed, time_taken, stats):
        """Update the processing progress and statistics"""
        total = stats['total_images']
        if total > 0:
            percentage = int((processed / total) * 100)
            self.process_progress_bar.setValue(percentage)
        else:
            self.process_progress_bar.setValue(0)
            
        self.process_progress_label.setText(f"Processing Images... ({processed}/{total})")
        self.total_images_label.setText(f"Total Images Processed: {total}")
        self.white_bg_label.setText(f"White Background Images: {stats['white_background']}")
        self.non_white_bg_label.setText(f"Non-White Background Images: {stats['non_white_background']}")
        self.time_label.setText(f"Time Taken: {time_taken}")
        
        extensions_text = ", ".join(f"{ext} ({count})" for ext, count in stats['extensions'].items())
        self.extensions_label.setText(f"Extensions: {extensions_text}")
    
    def processing_complete(self, stats):
        """Handle processing completion"""
        self.run_button.setEnabled(True)
        self.process_progress_bar.setValue(100)
        self.add_log("Processing complete! Excel report generated in the source folder.")
        
        # Show message box with results
        report_path = os.path.join(self.folder_input.text(), 'SplitImg_Report.xlsx')
        QMessageBox.information(self, "Processing Complete", 
                              f"Successfully processed {stats['total_images']} images.\n\n"
                              f"White background: {stats['white_background']}\n"
                              f"Non-white background: {stats['non_white_background']}\n\n"
                              f"Report saved to:\n{report_path}")
    
    def reset_application(self):
        """Reset the application to its initial state"""
        reply = QMessageBox.question(self, 'Reset SplitX',
                                   'Are you sure you want to reset? This will clear all inputs and progress.',
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                                   QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            # Reset inputs
            self.designer_input.clear()
            self.folder_input.clear()
            
            # Reset progress bars
            self.reset_progress()
            
            # Reset labels
            self.scan_progress_label.setText("Scanning Images...")
            self.process_progress_label.setText("Processing Images...")
            
            # Reset button state
            self.run_button.setEnabled(False)
            
            # Add log entry
            self.add_log("Application reset")
    
    def add_log(self, message):
        """Add a message to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def clear_log(self):
        """Clear the log"""
        self.log_text.clear()
        self.add_log("Log cleared") 