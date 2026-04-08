# AegisIRT: Multi-Language AI Self-Healing for Clinical Trial Systems

## Overview

AegisIRT is an advanced AI-powered self-healing system designed specifically for Interactive Voice Response Systems (IVRS) and Interactive Web Response Technologies (IRT) used in clinical trials. It leverages deep learning neural networks (TensorFlow.js for the frontend API and a Python backend for comprehensive training and model persistence) to automatically identify and suggest fixes for broken UI locators and elements.

The system is built to be highly adaptable, supporting a wide range of programming languages and testing frameworks, ensuring robust and reliable automation in complex clinical trial environments. With its feedback-driven learning mechanism, AegisIRT continuously improves its healing capabilities based on real-world user input.

## Features

*   **AI-Powered Locator Healing**: Utilizes a deep neural network to predict optimal healing strategies for broken UI elements.
*   **Multi-Language Compatibility**: Generates healing suggestions compatible with various programming languages and testing frameworks.
*   **Clinical Trial Context Awareness**: Incorporates specialized features and training data relevant to patient enrollment, randomization, drug dispensing, and other clinical trial workflows.
*   **Feedback-Driven Learning**: Allows users to approve, reject, or correct AI suggestions, continuously retraining and improving the underlying neural network.
*   **Real-time Predictions**: Provides immediate healing suggestions through a responsive web interface.
*   **Scalable Architecture**: Built with a Next.js frontend and a Python Flask backend, allowing for flexible deployment and integration.
*   **About Me Section**: Includes a dedicated section accessible via a button, providing insights into the creator's background, academic journey, passions, and professional aspirations, along with links to LinkedIn, Email, and GitHub.
*   **"A K.V.S.K Production" Footer**: A custom footer acknowledging the creator at the bottom of the application page.

## Supported Languages & Frameworks

AegisIRT's neural network is trained to generate compatible healing strategies for the following languages and their common testing frameworks:

*   **Java**: Selenium WebDriver
*   **C#**: Selenium WebDriver
*   **Python**: Selenium WebDriver
*   **C++**: Custom WebDriver
*   **SQL**: Database Testing
*   **JavaScript**: Playwright/WebDriverIO
*   **TypeScript**: Playwright
*   **Ruby**: Selenium WebDriver
*   **HTML**: DOM Testing
*   **CSS**: Style Testing
*   **YAML**: Config Testing
*   **Groovy**: Jenkins/Gradle
*   **Bash**: Shell Testing

## Architecture

AegisIRT consists of two main components:

1.  **Frontend (Next.js)**: A React-based web interface that allows users to input broken element details and HTML content. It communicates with the AI backend via Next.js API routes. It also hosts the "About Me" section and the application footer.
2.  **AI Backend (Python Flask with TensorFlow)**: A Python Flask server that hosts the core TensorFlow neural network. This server handles:
    *   **Feature Extraction**: Converts raw element and HTML data into numerical features.
    *   **Prediction**: Uses the trained neural network to predict healing probabilities and suggest new locators.
    *   **Training & Retraining**: Trains the initial model with comprehensive clinical trial data and continuously retrains it based on user feedback.
    *   **Model Persistence**: (Future enhancement) Saves and loads the trained model.

## Setup and Installation

To get AegisIRT up and running, follow these steps:

### Prerequisites

*   Node.js (LTS version recommended)
*   Python 3.8+
*   pip (Python package installer)

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd AegisIRT
