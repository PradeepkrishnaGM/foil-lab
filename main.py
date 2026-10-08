# main.py
import sys
from PySide6.QtWidgets import QApplication

# Requires the package to be installed (``pip install -e ".[gui]"``).
from foil_lab.gui.main_window import FoilLabMainWindow

def main():
    # Initialize the Qt Application
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Gives a clean, modern cross-platform look
    
    # Create and show the main window
    window = FoilLabMainWindow()
    window.show()
    
    # Run the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
