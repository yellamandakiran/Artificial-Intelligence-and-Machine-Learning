# Credit Card Approval Prediction System

> Artificial Intelligence & Machine Learning 

## 📌 Project Overview

The **Credit Card Approval Prediction System** is a Machine Learning and
Flask-based web application that predicts whether a credit card
application should be **Approved** or **Rejected** based on applicant
information.

The system automates the approval process using historical applicant and
credit data, reducing manual effort while improving decision-making
accuracy.
## 🚀 Features

-   User-friendly web interface
-   Applicant information input form
-   Data preprocessing and feature engineering
-   Multiple Machine Learning models
-   XGBoost selected as the best-performing model
-   Real-time prediction (Approved / Rejected)
-   Local deployment using Flask
-   Ready for cloud deployment
# Repository Structure

```
Artificial-Intelligence-and-Machine-Learning
│
├── 1. Brainstorming & Ideation
│   ├── Brainstorming & Idea Prioritization.pdf
│   ├── Define Problem Statements.pdf
│   └── Empathy Map.pdf
│
├── 2. Requirement Analysis
│   ├── Customer Journey Map.pdf
│   ├── Data Flow Diagram.pdf
│   ├── Solution Requirements.pdf
│   └── Technology Stack.pdf
│
├── 3. Project Design Phase
│   ├── Problem-Solution Fit.pdf
│   ├── Proposed Solution.pdf
│   └── Solution Architecture.pdf
│
├── 4. Project Planning Phase
│   └── Project Planning.pdf
│
├── 5. Project Development Phase
│   ├── Code-Layout, Readability and Reusability.pdf
│   ├── Coding & Solution.pdf
│   ├── No. of Functional Features Included in the Solution.pdf
│   │
│   └── CreditCardApprovalPrediction
│       ├── dataset
│       │   ├── application_record.csv
│       │   └── credit_record.csv
│       │
│       ├── static
│       │   ├── css
│       │   │   └── style.css
│       │   └── images
│       │
│       ├── templates
│       │   ├── index.html
│       │   └── result.html
│       │
│       ├── app.py
│       ├── train_model.py
│       ├── model.pkl
│       ├── label_encoders.pkl
│       ├── requirements.txt
│       └── README.md
│
├── 6. Project Testing
│   ├── Performance Testing.pdf
│   └── UAT Report.pdf
│
├── 7. Project Documentation
│   ├── Project Executable Files.pdf
│   └── Sample Project Documentation.pdf
│
├── 8. Project Demonstration
│   ├── Communication.pdf
│   ├── Demonstration of Proposed Features.pdf
│   ├── Project Demo Planning.pdf
│   ├── Scalability & Future Plan.pdf
│   └── Team Involvement in Demonstration.pdf
│
└── README.md
```
  
  ## 🛠 Technology Stack

### Frontend

-   HTML
-   CSS

### Backend

-   Python
-   Flask

### Machine Learning

-   Scikit-learn
-   XGBoost

### Data Processing

-   Pandas
-   NumPy

### Visualization

-   Matplotlib
-   Seaborn
  ## 📂 Dataset

Source: **Kaggle**

Files: - application_record.csv - credit_record.csv
## 🤖 Machine Learning Models

-   Logistic Regression
-   Decision Tree
-   Random Forest
-   XGBoost (Final Model)

------------------------------------------------------------------------

## ⚙️ Installation

### Clone Repository

``` bash
git clone https://github.com/yellamandakiran/Artificial-Intelligence-and-Machine-Learning.git
```

### Move to Project Folder

``` bash
cd Artificial-Intelligence-and-Machine-Learning/5.Project Development Phase/CreditCardApprovalPrediction
```

### Create Virtual Environment

``` bash
python -m venv venv
```

### Activate Virtual Environment (Windows)

``` bash
venv\Scripts\activate
```

### Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Train the Model

``` bash
python train_model.py
```

------------------------------------------------------------------------

## ▶️ Run the Application

``` bash
python app.py
```

Open your browser:

``` text
http://127.0.0.1:5000/
```

------------------------------------------------------------------------

## 🔄 Application Workflow

1.  Enter applicant details.
2.  Click **Predict**.
3.  Data is preprocessed.
4.  XGBoost predicts the result.
5.  The application displays **Approved** or **Rejected**.
