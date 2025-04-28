from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                           QLabel, QFileDialog, QProgressBar, QListWidget,
                           QHBoxLayout, QSplitter, QFrame, QTableWidget,
                           QTableWidgetItem, QComboBox, QCheckBox, QGroupBox,
                           QGridLayout, QHeaderView, QSpinBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage
import os
import time
from PIL import Image
import threading

def create_widget(parent=None):
    return FileSizeCheckerWidget(parent)

class ImageScannerWorker(QThread):
    progress_updated = pyqtSignal(int)
    file_processed = pyqtSignal(dict)
    scan_completed = pyqtSignal()
    error_occurred = pyqtSignal(str, str)
    
    def __init__(self, folder_path, min_size=0, max_size=None, include_subfolders=True):
        super().__init__()
        self.folder_path = folder_path
        self.min_size = min_size
        self.max_size = max_size
        self.include_subfolders = include_subfolders
        self.is_running = True
    
    def run(self):
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
        total_files = 0
        processed_files = 0
        
        # First count files for progress tracking
        for root, dirs, files in os.walk(self.folder_path):
            if not self.include_subfolders and root != self.folder_path:
                continue
                
            for file in files:
                if file.lower().endswith(image_extensions):
                    total_files += 1
        
        # Now process each file
        for root, dirs, files in os.walk(self.folder_path):
            if not self.is_running:
                break
                
            if not self.include_subfolders and root != self.folder_path:
                continue
                
            for file in files:
                if not self.is_running:
                    break
                    
                if file.lower().endswith(image_extensions):
                    file_path = os.path.join(root, file)
                    
                    try:
                        # Get file size
                        file_size = os.path.getsize(file_path)
                        
                        # Skip if outside size range
                        if (self.min_size and file_size < self.min_size * 1024) or \
                           (self.max_size and file_size > self.max_size * 1024):
                            processed_files += 1
                            self.progress_updated.emit(int(processed_files / total_files * 100))
                            continue
                        
                        # Get image dimensions
                        img = Image.open(file_path)
                        width, height = img.size
                        img.close()
                        
                        # Calculate relative path from search folder
                        rel_path = os.path.relpath(file_path, self.folder_path)
                        
                        # Emit file info
                        file_info = {
                            'filename': file,
                            'path': file_path,
                            'rel_path': rel_path,
                            'size': file_size,
                            'width': width,
                            'height': height,
                            'dimensions': f"{width}x{height}"
                        }
                        
                        self.file_processed.emit(file_info)
                        
                    except Exception as e:
                        self.error_occurred.emit(file_path, str(e))
                    
                    processed_files += 1
                    self.progress_updated.emit(int(processed_files / total_files * 100))
        
        self.scan_completed.emit()
    
    def stop(self):
        self.is_running = False


class ImagePreview(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(200)
        
        layout = QVBoxLayout(self)
        
        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(200)
        layout.addWidget(self.image_label)
        
        self.info_label = QLabel()
        layout.addWidget(self.info_label)
        
    def load_image(self, image_path):
        if not image_path or not os.path.exists(image_path):
            self.image_label.setText("Image not found")
            self.info_label.setText("")
            return
            
        try:
            # Get file info
            file_size = os.path.getsize(image_path)
            file_size_str = self.format_file_size(file_size)
            
            # Load image for preview
            pixmap = QPixmap(image_path)
            img = Image.open(image_path)
            
            self.info_label.setText(
                f"Path: {os.path.basename(image_path)}\n"
                f"Size: {file_size_str}\n"
                f"Dimensions: {img.width} x {img.height} px"
            )
            
            # Scale image to fit in preview while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_label.width(), 
                self.image_label.height(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.image_label.setText(f"Error: {str(e)}")
            self.info_label.setText("")
    
    def format_file_size(self, size_bytes):
        """Format file size in a human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"


class FileSizeCheckerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.files = []
        self.worker = None
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("File Size Checker")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # Controls area
        controls_group = QGroupBox("Scan Settings")
        controls_layout = QGridLayout(controls_group)
        
        # Folder selection
        self.folder_btn = QPushButton("Select Image Folder")
        self.folder_btn.clicked.connect(self.select_folder)
        controls_layout.addWidget(self.folder_btn, 0, 0)
        
        self.selected_folder = QLabel("No folder selected")
        controls_layout.addWidget(self.selected_folder, 0, 1, 1, 3)
        
        # Size filters
        controls_layout.addWidget(QLabel("Min Size (KB):"), 1, 0)
        self.min_size = QSpinBox()
        self.min_size.setRange(0, 1000000)
        self.min_size.setValue(0)
        controls_layout.addWidget(self.min_size, 1, 1)
        
        controls_layout.addWidget(QLabel("Max Size (KB):"), 1, 2)
        self.max_size = QSpinBox()
        self.max_size.setRange(0, 1000000)
        self.max_size.setValue(0)
        self.max_size.setSpecialValueText("No limit")
        controls_layout.addWidget(self.max_size, 1, 3)
        
        # Include subfolders
        self.include_subfolders = QCheckBox("Include Subfolders")
        self.include_subfolders.setChecked(True)
        controls_layout.addWidget(self.include_subfolders, 2, 0, 1, 2)
        
        # Scan button
        self.scan_btn = QPushButton("Scan Images")
        self.scan_btn.clicked.connect(self.scan_images)
        controls_layout.addWidget(self.scan_btn, 2, 2, 1, 2)
        
        main_layout.addWidget(controls_group)
        
        # Progress bar
        self.progress = QProgressBar()
        main_layout.addWidget(self.progress)
        
        # Main content area - split between results table and preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Results table
        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(["Filename", "Path", "Size", "Dimensions", "Actions"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.itemSelectionChanged.connect(self.on_selection_changed)
        splitter.addWidget(self.results_table)
        
        # Preview area
        self.preview = ImagePreview()
        splitter.addWidget(self.preview)
        
        # Set initial sizes (70% table, 30% preview)
        splitter.setSizes([int(self.width() * 0.7), int(self.width() * 0.3)])
        
        main_layout.addWidget(splitter, 1)  # Give it stretch
        
        # Status and filter area
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        self.sort_label = QLabel("Sort by:")
        status_layout.addWidget(self.sort_label)
        
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Filename", "Size (largest first)", "Size (smallest first)"])
        self.sort_combo.currentIndexChanged.connect(self.sort_results)
        status_layout.addWidget(self.sort_combo)
        
        self.export_btn = QPushButton("Export Report")
        self.export_btn.clicked.connect(self.export_report)
        status_layout.addWidget(self.export_btn)
        
        main_layout.addLayout(status_layout)
    
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.selected_folder.setText(folder)
            # Clear existing results
            self.results_table.setRowCount(0)
            self.files = []
            self.preview.image_label.setText("No image selected")
            self.preview.info_label.setText("")
    
    def scan_images(self):
        folder = self.selected_folder.text()
        if folder == "No folder selected":
            return
        
        # Get filter settings
        min_size = self.min_size.value()
        max_size = self.max_size.value() if self.max_size.value() > 0 else None
        include_subfolders = self.include_subfolders.isChecked()
        
        # Clear existing results
        self.results_table.setRowCount(0)
        self.files = []
        self.preview.image_label.setText("No image selected")
        self.preview.info_label.setText("")
        
        # Set up progress
        self.progress.setValue(0)
        
        # Disable controls during scan
        self.folder_btn.setEnabled(False)
        self.scan_btn.setEnabled(False)
        self.min_size.setEnabled(False)
        self.max_size.setEnabled(False)
        self.include_subfolders.setEnabled(False)
        
        # Start worker thread
        self.worker = ImageScannerWorker(folder, min_size, max_size, include_subfolders)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.file_processed.connect(self.add_file)
        self.worker.scan_completed.connect(self.scan_completed)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()
        
        self.status_label.setText("Scanning...")
    
    def update_progress(self, value):
        self.progress.setValue(value)
    
    def add_file(self, file_info):
        # Add to files list
        self.files.append(file_info)
        
        # Add to table
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # Set file name
        self.results_table.setItem(row, 0, QTableWidgetItem(file_info['filename']))
        
        # Set path (relative to scan folder)
        self.results_table.setItem(row, 1, QTableWidgetItem(file_info['rel_path']))
        
        # Set size
        size_str = self.format_file_size(file_info['size'])
        size_item = QTableWidgetItem(size_str)
        # Store actual size for sorting
        size_item.setData(Qt.ItemDataRole.UserRole, file_info['size'])
        self.results_table.setItem(row, 2, size_item)
        
        # Set dimensions
        self.results_table.setItem(row, 3, QTableWidgetItem(file_info['dimensions']))
        
        # Add view button
        view_button = QPushButton("View")
        view_button.setProperty("file_path", file_info['path'])
        view_button.clicked.connect(lambda: self.preview.load_image(self.sender().property("file_path")))
        self.results_table.setCellWidget(row, 4, view_button)
    
    def scan_completed(self):
        # Re-enable controls
        self.folder_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self.min_size.setEnabled(True)
        self.max_size.setEnabled(True)
        self.include_subfolders.setEnabled(True)
        
        # Update status
        self.status_label.setText(f"Found {len(self.files)} images")
        
        # Apply initial sort
        self.sort_results()
    
    def handle_error(self, file_path, error_msg):
        print(f"Error processing {file_path}: {error_msg}")
    
    def on_selection_changed(self):
        selected_items = self.results_table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            file_path = self.files[row]['path']
            self.preview.load_image(file_path)
    
    def sort_results(self):
        sort_index = self.sort_combo.currentIndex()
        
        if sort_index == 0:  # Filename
            self.files.sort(key=lambda x: x['filename'])
        elif sort_index == 1:  # Size (largest first)
            self.files.sort(key=lambda x: x['size'], reverse=True)
        elif sort_index == 2:  # Size (smallest first)
            self.files.sort(key=lambda x: x['size'])
        
        # Rebuild table with sorted data
        self.results_table.setRowCount(0)
        for file_info in self.files:
            self.add_file(file_info)
    
    def export_report(self):
        if not self.files:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", "", "CSV Files (*.csv);;Text Files (*.txt)"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w') as f:
                # Write header
                f.write("Filename,Path,Size (bytes),Size (formatted),Width,Height\n")
                
                # Write data
                for file_info in self.files:
                    size_str = self.format_file_size(file_info['size'])
                    f.write(f"{file_info['filename']},{file_info['rel_path']},{file_info['size']},"
                           f"{size_str},{file_info['width']},{file_info['height']}\n")
                           
            self.status_label.setText(f"Report exported to {file_path}")
        except Exception as e:
            self.status_label.setText(f"Error exporting report: {str(e)}")
    
    def format_file_size(self, size_bytes):
        """Format file size in a human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
