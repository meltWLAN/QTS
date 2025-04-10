#!/usr/bin/env python3
"""
Quantum Symbiotic Network - Desktop Entry Point
This is the main entry point for the desktop version of the Quantum Symbiotic Network.
"""

import sys
import os
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from src.ui.desktop.main_window import MainWindow
from src.core.quantum_engine import QuantumEngine
from src.core.module_symbiosis_manager import ModuleSymbiosisManager
from src.utils.logger import setup_logger

def main():
    """Main entry point for the desktop application."""
    # Setup logging
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Starting Quantum Symbiotic Network Desktop Application")

    try:
        # Initialize core components
        quantum_engine = QuantumEngine()
        symbiosis_manager = ModuleSymbiosisManager(quantum_engine)

        # Create Qt application
        app = QApplication(sys.argv)
        app.setApplicationName("Quantum Symbiotic Network")
        app.setStyle("Fusion")  # Use Fusion style for a modern look

        # Create and show main window
        window = MainWindow(symbiosis_manager)
        window.show()

        # Start the event loop
        sys.exit(app.exec_())

    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 