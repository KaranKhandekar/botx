from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QOperatingSystemVersion, QSysInfo

def get_system_theme():
    """Detect if system is using dark mode (macOS specific)"""
    try:
        # This is a simplified check - on macOS we would use AppKit
        # For demonstration purposes only
        if QSysInfo.productType() == "macos":
            from Foundation import NSUserDefaults
            return "Dark" if NSUserDefaults.standardUserDefaults().stringForKey_("AppleInterfaceStyle") == "Dark" else "Light"
    except Exception:
        pass
    
    # Default to dark for professional look
    return "Dark"

# Professional dark theme with modern styling
DARK_THEME = """
    /* Base Elements */
    QWidget {
        background-color: #1E1E1E;
        color: #E8E8E8;
        font-family: 'SF Pro Display', -apple-system, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        font-size: 13px;
    }
    
    /* Window and Frames */
    QMainWindow {
        background-color: #252526;
    }
    
    QDialog {
        background-color: #252526;
        border: 1px solid #3C3C3C;
        border-radius: 8px;
    }
    
    QFrame {
        border-radius: 8px;
    }
    
    /* Text Elements */
    QLabel {
        color: #E8E8E8;
        padding: 2px;
    }
    
    QLabel[title="true"] {
        font-size: 22px;
        font-weight: bold;
        color: #3C99DC;
        margin-bottom: 12px;
    }
    
    QLabel[subtitle="true"] {
        font-size: 16px;
        color: #BBBBBB;
    }
    
    /* Buttons */
    QPushButton {
        background-color: #3C3C3C;
        color: #E8E8E8;
        border: none;
        border-radius: 6px;
        padding: 10px 18px;
        font-weight: 500;
        min-height: 22px;
    }
    
    QPushButton:hover {
        background-color: #505050;
    }
    
    QPushButton:pressed {
        background-color: #2A2A2A;
    }
    
    QPushButton:disabled {
        background-color: #2A2A2A;
        color: #666666;
    }
    
    QPushButton[primary="true"] {
        background-color: #0078D4;
        color: white;
    }
    
    QPushButton[primary="true"]:hover {
        background-color: #0086F0;
    }
    
    QPushButton[primary="true"]:pressed {
        background-color: #006AC1;
    }
    
    QPushButton[secondary="true"] {
        background-color: transparent;
        color: #3C99DC;
        border: 1px solid #3C99DC;
    }
    
    QPushButton[secondary="true"]:hover {
        background-color: rgba(60, 153, 220, 0.1);
    }
    
    QPushButton[destructive="true"] {
        background-color: #D32F2F;
    }
    
    QPushButton[destructive="true"]:hover {
        background-color: #E53935;
    }
    
    /* Input Fields */
    QLineEdit, QTextEdit, QPlainTextEdit {
        border: 1px solid #3C3C3C;
        border-radius: 6px;
        padding: 10px;
        background-color: #2D2D30;
        selection-background-color: #264F78;
        color: #E8E8E8;
    }
    
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
        border: 2px solid #0078D4;
        padding: 9px;
    }
    
    /* Dropdown and List Elements */
    QComboBox {
        border: 1px solid #3C3C3C;
        border-radius: 6px;
        padding: 8px 14px;
        background-color: #2D2D30;
        min-height: 22px;
        color: #E8E8E8;
    }
    
    QComboBox:hover {
        border-color: #0078D4;
    }
    
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: center right;
        width: 24px;
        border-left: none;
    }
    
    QListWidget, QTreeView, QTableView {
        border: 1px solid #3C3C3C;
        border-radius: 6px;
        background-color: #2D2D30;
        alternate-background-color: #252526;
        color: #E8E8E8;
    }
    
    QListWidget::item, QTreeView::item, QTableView::item {
        padding: 6px;
        border-radius: 4px;
    }
    
    QListWidget::item:selected, QTreeView::item:selected, QTableView::item:selected {
        background-color: #3C3C3C;
        color: #FFFFFF;
    }
    
    QListWidget::item:hover, QTreeView::item:hover, QTableView::item:hover {
        background-color: #333333;
    }
    
    /* Tabs */
    QTabWidget::pane {
        border: 1px solid #3C3C3C;
        border-radius: 6px;
        top: -1px;
    }
    
    QTabBar::tab {
        background-color: #252526;
        border: 1px solid #3C3C3C;
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        padding: 10px 18px;
        margin-right: 2px;
    }
    
    QTabBar::tab:selected {
        background-color: #2D2D30;
        border-bottom: 2px solid #0078D4;
    }
    
    /* Group Box */
    QGroupBox {
        background-color: #252526;
        border: 1px solid #3C3C3C;
        border-radius: 8px;
        margin-top: 20px;
        font-weight: bold;
        padding: 15px;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 15px;
        color: #BBBBBB;
        padding: 0 5px;
    }
    
    /* Progress Bar */
    QProgressBar {
        border: none;
        border-radius: 4px;
        background-color: #3C3C3C;
        text-align: center;
        color: white;
        height: 10px;
    }
    
    QProgressBar::chunk {
        background-color: #0078D4;
        border-radius: 4px;
    }
    
    /* Scroll Bars */
    QScrollBar:vertical {
        background-color: #1E1E1E;
        width: 14px;
        margin: 0px;
        border-radius: 7px;
    }
    
    QScrollBar::handle:vertical {
        background-color: #3C3C3C;
        min-height: 30px;
        border-radius: 7px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #505050;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    
    QScrollBar:horizontal {
        background-color: #1E1E1E;
        height: 14px;
        margin: 0px;
        border-radius: 7px;
    }
    
    QScrollBar::handle:horizontal {
        background-color: #3C3C3C;
        min-width: 30px;
        border-radius: 7px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background-color: #505050;
    }
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }
    
    /* Radio Buttons */
    QRadioButton {
        spacing: 10px;
        color: #E8E8E8;
    }
    
    QRadioButton::indicator {
        width: 18px;
        height: 18px;
    }
    
    QRadioButton::indicator:checked {
        color: #0078D4;
    }
    
    /* Checkbox */
    QCheckBox {
        spacing: 10px;
        color: #E8E8E8;
    }
    
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border: 1px solid #505050;
        border-radius: 4px;
    }
    
    QCheckBox::indicator:checked {
        background-color: #0078D4;
        border-color: #0078D4;
    }
    
    /* Sidebar styling */
    QWidget#sidebar {
        background-color: #1E1E1E;
        border-right: 1px solid #333333;
    }
    
    QWidget#sidebarButton {
        text-align: left;
        padding-left: 20px;
        border-radius: 0;
        border-left: 5px solid transparent;
        background-color: transparent;
        height: 45px;
    }
    
    QWidget#sidebarButton:hover {
        background-color: #2A2A2A;
        border-left: 5px solid #505050;
    }
    
    QWidget#sidebarButton[active="true"] {
        background-color: #252526;
        border-left: 5px solid #0078D4;
    }
    
    QWidget#sidebarHeader {
        padding: 20px;
        border-bottom: 1px solid #333333;
    }
    
    /* Splitter */
    QSplitter::handle {
        background-color: #333333;
        height: 1px;
        width: 1px;
    }
    
    QSplitter::handle:hover {
        background-color: #0078D4;
    }
    
    /* Status Bar */
    QStatusBar {
        background-color: #1E1E1E;
        color: #BBBBBB;
        border-top: 1px solid #333333;
    }
    
    /* Menu */
    QMenuBar {
        background-color: #1E1E1E;
        border-bottom: 1px solid #333333;
    }
    
    QMenuBar::item {
        padding: 8px 15px;
        background: transparent;
    }
    
    QMenuBar::item:selected {
        background-color: #333333;
        border-radius: 4px;
    }
    
    QMenu {
        background-color: #252526;
        border: 1px solid #3C3C3C;
        border-radius: 6px;
    }
    
    QMenu::item {
        padding: 8px 30px 8px 20px;
    }
    
    QMenu::item:selected {
        background-color: #0078D4;
        color: white;
    }
    
    /* Tooltip */
    QToolTip {
        border: 1px solid #505050;
        border-radius: 4px;
        background-color: #252526;
        color: #E8E8E8;
        padding: 5px;
    }
"""

# Light theme with similar styling to dark theme for consistency
LIGHT_THEME = """
    /* Base Elements */
    QWidget {
        background-color: #FFFFFF;
        color: #333333;
        font-family: 'SF Pro Display', -apple-system, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        font-size: 13px;
    }
    
    /* Window and Frames */
    QMainWindow {
        background-color: #F5F5F5;
    }
    
    QDialog {
        background-color: #FFFFFF;
        border: 1px solid #DADADA;
        border-radius: 8px;
    }
    
    QFrame {
        border-radius: 8px;
    }
    
    /* Text Elements */
    QLabel {
        color: #333333;
        padding: 2px;
    }
    
    QLabel[title="true"] {
        font-size: 22px;
        font-weight: bold;
        color: #0078D4;
        margin-bottom: 12px;
    }
    
    QLabel[subtitle="true"] {
        font-size: 16px;
        color: #666666;
    }
    
    /* Buttons */
    QPushButton {
        background-color: #333333;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 18px;
        font-weight: 500;
        min-height: 22px;
    }
    
    QPushButton:hover {
        background-color: #505050;
    }
    
    QPushButton:pressed {
        background-color: #2A2A2A;
    }
    
    QPushButton:disabled {
        background-color: #2A2A2A;
        color: #666666;
    }
    
    QPushButton[primary="true"] {
        background-color: #0078D4;
        color: white;
    }
    
    QPushButton[primary="true"]:hover {
        background-color: #0086F0;
    }
    
    QPushButton[primary="true"]:pressed {
        background-color: #006AC1;
    }
    
    QPushButton[secondary="true"] {
        background-color: transparent;
        color: #333333;
        border: 1px solid #333333;
    }
    
    QPushButton[secondary="true"]:hover {
        background-color: rgba(51, 51, 51, 0.1);
    }
    
    QPushButton[destructive="true"] {
        background-color: #D32F2F;
    }
    
    QPushButton[destructive="true"]:hover {
        background-color: #E53935;
    }
    
    /* Input Fields */
    QLineEdit, QTextEdit, QPlainTextEdit {
        border: 1px solid #333333;
        border-radius: 6px;
        padding: 10px;
        background-color: #2D2D30;
        selection-background-color: #264F78;
        color: white;
    }
    
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
        border: 2px solid #0078D4;
        padding: 9px;
    }
    
    /* Dropdown and List Elements */
    QComboBox {
        border: 1px solid #333333;
        border-radius: 6px;
        padding: 8px 14px;
        background-color: #2D2D30;
        min-height: 22px;
        color: white;
    }
    
    QComboBox:hover {
        border-color: #0078D4;
    }
    
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: center right;
        width: 24px;
        border-left: none;
    }
    
    QListWidget, QTreeView, QTableView {
        border: 1px solid #333333;
        border-radius: 6px;
        background-color: #2D2D30;
        alternate-background-color: #252526;
        color: white;
    }
    
    QListWidget::item, QTreeView::item, QTableView::item {
        padding: 6px;
        border-radius: 4px;
    }
    
    QListWidget::item:selected, QTreeView::item:selected, QTableView::item:selected {
        background-color: #333333;
        color: white;
    }
    
    QListWidget::item:hover, QTreeView::item:hover, QTableView::item:hover {
        background-color: #333333;
    }
    
    /* Tabs */
    QTabWidget::pane {
        border: 1px solid #333333;
        border-radius: 6px;
        top: -1px;
    }
    
    QTabBar::tab {
        background-color: #252526;
        border: 1px solid #333333;
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        padding: 10px 18px;
        margin-right: 2px;
    }
    
    QTabBar::tab:selected {
        background-color: #2D2D30;
        border-bottom: 2px solid #0078D4;
    }
    
    /* Group Box */
    QGroupBox {
        background-color: #252526;
        border: 1px solid #333333;
        border-radius: 8px;
        margin-top: 20px;
        font-weight: bold;
        padding: 15px;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 15px;
        color: #BBBBBB;
        padding: 0 5px;
    }
    
    /* Progress Bar */
    QProgressBar {
        border: none;
        border-radius: 4px;
        background-color: #333333;
        text-align: center;
        color: white;
        height: 10px;
    }
    
    QProgressBar::chunk {
        background-color: #0078D4;
        border-radius: 4px;
    }
    
    /* Scroll Bars */
    QScrollBar:vertical {
        background-color: #1E1E1E;
        width: 14px;
        margin: 0px;
        border-radius: 7px;
    }
    
    QScrollBar::handle:vertical {
        background-color: #333333;
        min-height: 30px;
        border-radius: 7px;
    }
    
    QScrollBar::handle:vertical:hover {
        background-color: #505050;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    
    QScrollBar:horizontal {
        background-color: #1E1E1E;
        height: 14px;
        margin: 0px;
        border-radius: 7px;
    }
    
    QScrollBar::handle:horizontal {
        background-color: #333333;
        min-width: 30px;
        border-radius: 7px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background-color: #505050;
    }
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }
    
    /* Radio Buttons */
    QRadioButton {
        spacing: 10px;
        color: white;
    }
    
    QRadioButton::indicator {
        width: 18px;
        height: 18px;
    }
    
    QRadioButton::indicator:checked {
        color: #0078D4;
    }
    
    /* Checkbox */
    QCheckBox {
        spacing: 10px;
        color: white;
    }
    
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border: 1px solid #505050;
        border-radius: 4px;
    }
    
    QCheckBox::indicator:checked {
        background-color: #0078D4;
        border-color: #0078D4;
    }
    
    /* Sidebar styling */
    QWidget#sidebar {
        background-color: #1E1E1E;
        border-right: 1px solid #333333;
    }
    
    QWidget#sidebarButton {
        text-align: left;
        padding-left: 20px;
        border-radius: 0;
        border-left: 5px solid transparent;
        background-color: transparent;
        height: 45px;
    }
    
    QWidget#sidebarButton:hover {
        background-color: #2A2A2A;
        border-left: 5px solid #505050;
    }
    
    QWidget#sidebarButton[active="true"] {
        background-color: #252526;
        border-left: 5px solid #0078D4;
    }
    
    QWidget#sidebarHeader {
        padding: 20px;
        border-bottom: 1px solid #333333;
    }
    
    /* Splitter */
    QSplitter::handle {
        background-color: #333333;
        height: 1px;
        width: 1px;
    }
    
    QSplitter::handle:hover {
        background-color: #0078D4;
    }
    
    /* Status Bar */
    QStatusBar {
        background-color: #1E1E1E;
        color: #BBBBBB;
        border-top: 1px solid #333333;
    }
    
    /* Menu */
    QMenuBar {
        background-color: #1E1E1E;
        border-bottom: 1px solid #333333;
    }
    
    QMenuBar::item {
        padding: 8px 15px;
        background: transparent;
    }
    
    QMenuBar::item:selected {
        background-color: #333333;
        border-radius: 4px;
    }
    
    QMenu {
        background-color: #252526;
        border: 1px solid #333333;
        border-radius: 6px;
    }
    
    QMenu::item {
        padding: 8px 30px 8px 20px;
    }
    
    QMenu::item:selected {
        background-color: #0078D4;
        color: white;
    }
    
    /* Tooltip */
    QToolTip {
        border: 1px solid #505050;
        border-radius: 4px;
        background-color: #252526;
        color: #E8E8E8;
        padding: 5px;
    }
"""

def apply_theme(window, theme_name):
    """Apply the specified theme to the window"""
    if theme_name == "Auto (System)":
        theme_name = get_system_theme()
        
    if theme_name == "Dark":
        window.setStyleSheet(DARK_THEME)
    else:  # Light theme
        window.setStyleSheet(LIGHT_THEME)
