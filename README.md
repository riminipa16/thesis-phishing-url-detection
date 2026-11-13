# Phishing URL Detection System

A machine learning-based phishing URL detection system featuring a Python Flask REST API backend and a browser extension frontend for real-time analysis. Developed as part of an academic thesis project.

## Project Overview
Phishing attacks remain one of the most prevalent cybersecurity threats. This project provides an end-to-end detection pipeline:
1. **Machine Learning Pipeline:** Trained on URL/email phishing features to classify suspicious links.
2. **Backend API (`app.py`):** A lightweight Flask server that exposes a `/predict` endpoint for real-time inference.
3. **Browser Extension (`extension/`):** A Manifest V3 browser extension allowing users to submit URLs or text snippets directly from their browser for immediate classification.



## Dataset
This project uses the **CEAS_08** dataset. 
1. Download the dataset from Zenodo: [Zenodo Record 8339691](https://zenodo.org/records/8339691).
2. Download/extract `CEAS_08.csv` and place it inside a folder named `data/` in the root directory.
