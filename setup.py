from setuptools import setup, find_packages

setup(
    name="height_estimator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.0",
        "numpy>=1.19.0",
        "torch>=1.9.0",
        "transformers>=4.0.0",
        "streamlit>=1.0.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered height estimation system"
)