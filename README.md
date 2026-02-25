# CPR Framework
A lightweight and efficient Cycle and Phase Recovery (CPR) framework for periodic time series reconstruction, with privacy protection capabilities based on differential privacy.

## 🚀 Quick Start
### 1. Core Objective
This guide provides a streamlined workflow for getting started with the CPR Framework, focusing on the core stages of "Environment Setup - Data Preparation - Code Execution - Result Validation".

### 2. Prerequisites (Mandatory)
#### 2.1 Basic Environment
- Python 3.7 ~ 3.10 (Python 3.8 is recommended for optimal compatibility)

#### 2.2 Installation of Core Libraries
Copy and execute the following command in the terminal/command prompt to install all dependencies in one click:
```bash
pip install pandas numpy matplotlib scikit-learn scipy openpyxl
```

### 3. File Preparation
#### 3.1 Required Files
1. Code file: `main.py` (core implementation of the CPR Framework)
2. Dataset files (select any one for testing):
   - `crowdsourced+mapping.xlsx` (NDVI periodic data)
   - `darwin.xlsx` (handwriting sample periodic data)
   - `raisin.xlsx` (raisin feature periodic data)
   - `turkish+music+emotion.xlsx` (Turkish music emotion periodic data)

### 4. Quick Execution Steps
#### 4.1 Step 1: Modify Code Parameters (Only 3 Key Configurations)
Open `main.py` with a text editor, locate the parameter configuration section, and modify the following parameters:
```python
# 1. Dataset path (replace with your actual dataset path)
DATA_PATH = "required_dataset.xlsx"

# 2. Privacy budget (balance between privacy and data utility)
EPSILON = 2.0

# 3. Window size (fixed as 5 for beginners)
WINDOW_SIZE = 5
```

#### 4.2 Step 2: Run the Code
Execute the following command in the terminal (under the project root directory):
```bash
python main.py
```
