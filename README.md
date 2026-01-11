# 💡 Book Scanner  
Automated Video-Based Book Digitization

<!-- Optional: add a project cover image -->
<!-- ![Project Cover](media/cover.png) -->

---

## Table of Contents
- [The Team](#-the-team)
- [Project Description](#-project-description)
- [Getting Started](#-getting-started)
  - [Prerequisites](#-prerequisites)
  - [Installing](#-installing)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Built With](#-built-with)
- [Acknowledgments](#-acknowledgments)

---

## 👥 The Team

**Project Author**  
- Yair Sheffer  
- GitHub: https://github.com/yairsheffer1  

**Advisor & Mentor**  
- Gal Katzhendler  
- The Hebrew University of Jerusalem
---

## 📚 Project Description

This project presents a fully automated, software-only pipeline for digitizing printed books directly from handheld video recordings.

Instead of relying on specialized scanning hardware, the system processes a continuous video of page flipping and automatically produces a clean, ordered, and searchable PDF document.

### Main Features
- Fully automated end-to-end pipeline
- No dedicated scanning hardware required
- Robust handling of page curvature, background clutter, and uneven illumination
- Produces OCR-enabled searchable PDFs
- Modular design enabling independent testing and extension

### Main Components
- Frame extraction and stability detection  
- Page splitting and grouping  
- Corner-based cropping  
- Masking and image enhancement  
- Geometric dewarping  
- PDF assembly and OCR  

### Technologies Used
- Python
- OpenCV
- Tesseract OCR
- NumPy
- Matplotlib

---

## ⚡ Getting Started

These instructions will help you set up and run the project locally for development and testing.

---

## 🧱 Prerequisites

Make sure the following tools are installed:

- Python 3.8 or higher
- pip (Python package manager)
- Tesseract OCR

Example:
```bash
sudo apt install tesseract-ocr
