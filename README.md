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


---

## 🧪 Testing

This project focuses on end-to-end functional validation rather than unit-level testing.

Testing the system involves:
- Providing a short video of a book being flipped
- Running the full processing pipeline
- Verifying that a clean, ordered, and searchable PDF is produced

### Sample Tests
- **Frame Stability Test**: Verifies that selected frames correspond to stable, non-moving pages.
- **Page Splitting Test**: Ensures correct left/right page separation.
- **Dewarping Test**: Confirms that curved pages are rectified without damaging text content.
- **OCR Validation**: Compares extracted text against reference digital scans for accuracy.

These tests collectively ensure that each stage of the pipeline performs reliably under intended capture conditions.

---

## 🚀 Deployment

The system is designed for offline, local execution and does not require server-side deployment.

Typical deployment workflow:
1. Capture a video of page flipping using a standard smartphone.
2. Transfer the video to a local machine.
3. Run the pipeline to generate a searchable PDF.

The pipeline operates fully automatically after video capture and requires no further user interaction.

---

## ⚙️ Built With

This project builds upon and integrates the following technologies and tools:

- **Python** – Core implementation language
- **OpenCV** – Image and video processing
- **Tesseract OCR** – Optical character recognition
- **Zucker Page Dewarping** – Geometric page rectification
- **NumPy & Matplotlib** – Numerical processing and visualization

---

## 🙏 Acknowledgments

- M. Zucker for the open-source page dewarping implementation
- The Tesseract OCR project and contributors
- Course staff and advisor for guidance and feedback
- Open-source community for providing essential tools and inspiration

