from cx_Freeze import setup, Executable
import sys

# Dependencies are automatically detected, but it might need fine tuning.
build_options = {
    'packages': [
        'sklearn',
        'pandas', 
        'numpy',
        'matplotlib',
        'seaborn',
        'warnings',
        'scipy'  # sklearn dependency
    ],
    'excludes': [
        'tkinter',  # Exclude GUI libraries if not needed
        'PyQt5',
        'PyQt6'
    ],
    'include_files': [
        'bank_transactions_enhanced_fraud_detection.csv',  # Your actual CSV file
        # Add any other data files your application needs
    ],

    'optimize': 2,  # Optimize bytecode
    'zip_include_packages': ['*'],  # Package everything in ZIP for faster startup
    'zip_exclude_packages': []
}

# Base configuration for different platforms
base = None
if sys.platform == "win32":
    base = "Console"  # Use "Win32GUI" for windowed applications without console

# Define the executable
executables = [
    Executable(
        "fraud_detection.py",  # Your main Python file
        base=base,
        target_name="FraudDetectionSystem.exe",  # Name of the executable
        icon=None,  
        copyright="Copyright (C) 2025 Your Company"
    )
]

setup(
    name="FraudDetectionSystem",
    version="1.0.0",
    description="Advanced Fraud Detection System using Machine Learning",
    long_description="An Isolation Forest-based fraud detection system for financial transactions",
    
    options={'build_exe': build_options},
    executables=executables,
    
    # Additional metadata
    license="MIT",
    keywords="fraud detection machine learning isolation forest",
)