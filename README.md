# 🛡️ PhishGuard AI — Intelligent Phishing Detection System

**Detect Deception. Browse with Confidence.** — An enterprise-grade, AI-powered URL threat detection platform built with Python, Streamlit, Scikit-learn, and Pandas.

---

## 📌 Project Overview

**PhishGuard AI** is a modern cybersecurity web application that helps users and security analysts detect and understand phishing threats. Operating strictly in an offline, zero-egress sandbox, it extracts 18 static lexical and structural features from website URLs, executes an ensemble Random Forest classifier, calculates a calibrated 0–100 risk score, and provides actionable defensive safety recommendations.

---

## ✨ Key Capabilities

* **🏠 Homepage Hero Section:** "Think Before You Click" hero header with quick actions and a clear privacy guarantee.
* **🔍 Main AI URL Scanner:** Single-click lexical threat detection with circular progress gauges, ML probabilities, and heuristic risk scores.
* **🧬 URL Anatomy Dissection:** Visual breakdown separating protocol, hostname, subdomain, domain, path, query, and port with syntax pills.
* **📊 Interactive Cybersecurity Dashboard:** Real-time session KPI cards (Total Scans, High-Risk URLs, Low-Risk URLs, Accuracy) paired with donut risk distribution charts and scan activity timelines.
* **⚡ Batch Phishing Detection:** Process dozens of URLs concurrently by pasting raw text or uploading CSV files, with category filters and CSV export.
* **🧠 Explainable AI:** Clear decomposition of machine learning predictions vs. heuristic risk scores and structural rule triggers.
* **🛡️ Phishing Awareness Portal:** Educational guidance on typosquatting, subdomain disguises, URL shorteners, and the HTTPS misconception.
* **⚙️ Model Diagnostics & Evaluation:** Real confusion matrix and Gini feature importance rankings from a held-out test split.

---

## 🛠️ Technology Stack

* **Application Framework:** Streamlit
* **Machine Learning:** Scikit-learn (`RandomForestClassifier`, `train_test_split`, `metrics`)
* **Data Manipulation:** Pandas, NumPy
* **Visualizations:** Matplotlib, Seaborn
* **Language:** Python 3.10+
* **Architecture:** Single-file self-contained application (`app.py`)

---

## 📂 Required Project Files

The project conforms strictly to the four-file specification:

```text
phishing_detection/
│
├── app.py              # Complete application code, ML pipeline, and custom SaaS CSS
├── requirements.txt    # Minimal pinned Python dependencies
├── README.md           # Product documentation and execution guide
└── report.md           # Comprehensive academic college project report
```

---

## ⚙️ Installation & Setup Instructions

### Step 1: Open VS Code Terminal

Open Visual Studio Code, navigate to **File > Open Folder...**, and select the `phishing_detection` directory. Open the integrated terminal (`Ctrl + ` `).

### Step 2: Create a Virtual Environment (Recommended)

**Windows PowerShell:**

```powershell
python -m venv .venv
```

**macOS / Linux:**

```bash
python3 -m venv .venv
```

### Step 3: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

*(If script execution is restricted on Windows, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Launch the web interface by running:

```bash
streamlit run app.py
```

The application will launch and open in your default web browser at:

```text
http://localhost:8501
```

---

## 🔒 Security and Privacy Guarantees

* **Zero Remote Connections:** PhishGuard AI never establishes HTTP connections or DNS queries to target URLs.
* **Zero Code Execution:** Remote scripts, cookies, stylesheets, and payloads are never fetched or executed.
* **No Permanent Tracking:** User queries are held in volatile session memory by default and can be cleared with one click.
* **Educational Disclaimer:** Algorithmic risk scores are statistical estimates and do not constitute an absolute guarantee of website safety.
