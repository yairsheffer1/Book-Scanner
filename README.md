# 💡 Book Scanner  
Automated Video-Based Book Digitization

![Project projectLogo](media/projectLogo.png)

---

## Table of Contents
- [The Team](#the-team)
- [Project Description](#project-description)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installing](#installing)
- [Testing](#testing)
- [Deployment](#deployment)
- [Built With](#built-with)
- [Acknowledgments](#acknowledgments)

---

## 👥 The Team

**Project Author**  
- [Member 1](https://github.com/yairsheffer1)

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

## 🏗️ Installing

Install the required Python dependencies using pip:

```bash
pip install -r requirements.txt
```

---

## 🧪 Testing

This project emphasizes pipeline-level functional validation with explicit inspection
of intermediate outputs, rather than relying solely on black-box end-to-end testing.

The system is evaluated by running real capture data through the full processing
pipeline, while persisting debug artifacts for each processing stage. Each stage
produces a dedicated output directory, allowing independent verification of correctness
before progressing to the next step.

The validation workflow includes:

• Supplying a short video of a book being flipped
• Executing the full processing pipeline
• Persisting intermediate results for every major stage
• Validating each stage independently using its debug outputs
• Verifying that the final result is a clean, ordered, and searchable PDF

Pipeline Stage Validation:

**Frame Selecting:**
Verifies that only stable, non-moving frames are selected as candidate pages.

**Page Splitting:**
Validates correct separation of left and right pages, using saved split images.

**Corner-Based Cropping:**
Verifies safe and consistent background reduction with no observed content loss.

**Grouping and Ordering:**
Confirms that pages are grouped and ordered correctly prior to PDF generation.

**Dewarping:**
Ensures curved pages are geometrically rectified without introducing text distortion,
by inspecting both original and cleaned dewarped outputs.

**OCR Validation:**
Evaluates extracted text accuracy by comparing OCR output against reference digital scans.

This structured, stage-by-stage validation approach enables precise debugging,
isolates failures to specific pipeline components, and ensures robustness under
realistic capture conditions.

---

## 🚀 Deployment

The system is designed for offline, local execution and runs directly on a user’s
machine within the configured development environment.

Deployment consists of running the processing pipeline as a Python-based application,
where the input video path and the expected number of pages are provided as runtime
parameters.

Typical deployment workflow:

• Capture a short video of page flipping using a standard smartphone
• Transfer the video file to a local machine
• Configure the input parameters (video path and expected page count)
• Execute the full processing pipeline to generate a clean, ordered, and searchable PDF

Once executed, the pipeline runs fully automatically and requires no further user
interaction beyond the initial parameter configuration.

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

