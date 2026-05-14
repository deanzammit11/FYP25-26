# FYP25-26 Setup Guide
This guide explains how to set up and run the project locally.

## Prerequisites
Ensure the following software is installed on your machine:
- Python 3.10 or newer
- Visual Studio Code (recommended)

### Downloads
- Python: https://www.python.org/downloads/
- VS Code: https://code.visualstudio.com/

## 1. Open the Project in VS Code
Open the `FYP25-26` project folder in Visual Studio Code.

You can do this by selecting:
**File → Open Folder → FYP25-26**

Ensure that the terminal path is inside the project directory and ends with `FYP25-26`.

## 2. Create a Virtual Environment
Create a Python virtual environment to isolate the project dependencies:
```bash
python -m venv venv
```

## 3. Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

Once activated, `(venv)` should appear in the terminal.

## 4. Install Project Dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
```

## 5. Run the Machine Learning Pipeline
Start the application by running:
```bash
python main.py
```
This command executes the machine learning pipeline.

## Project Structure
```
FYP25-26/
│
├── data/
├── src/
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

## Notes
- Ensure the virtual environment is activated before running the project.
- If additional dependencies are required, install them using:
```bash
  pip install <package-name>
```
- If Python is not recognised, try using `py main.py` instead.