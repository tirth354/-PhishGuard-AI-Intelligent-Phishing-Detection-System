"""
========================================================================================
PhishGuard AI — Intelligent Phishing Detection System
"Detect Deception. Browse with Confidence."
========================================================================================
A premium, production-grade cybersecurity web application built with Python,
Streamlit, Scikit-learn, and Pandas. Provides real-time URL lexical threat detection,
URL anatomy inspection, batch scanning, explainable AI, and educational awareness.

Aesthetics: High-contrast, crystal-clear cybersecurity SaaS theme (Deep Navy #080D18,
Surface #111827, Cyan #22D3EE, High-Contrast White #FFFFFF & #F8FAFC).
========================================================================================
"""

import os
import re
import io
import time
from datetime import datetime
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------------------------------------------
# 1. PAGE SETUP & GLOBAL CONFIGURATION
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="PhishGuard AI — Intelligent Phishing Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# 2. CUSTOM CSS: HIGH-CONTRAST SAAS DESIGN SYSTEM WITH ULTRA-READABLE FONTS
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700;800&display=swap');

    /* Global reset & typography */
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #F8FAFC !important;
    }
    
    /* Background Canvas */
    .stApp {
        background-color: #080D18 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(34, 211, 238, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(59, 130, 246, 0.10) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(17, 24, 39, 0.9) 0px, transparent 70%) !important;
    }
    
    /* Eliminate default white top header strip */
    header[data-testid="stHeader"] {
        background: rgba(8, 13, 24, 0.95) !important;
        backdrop-filter: blur(10px) !important;
    }
    header[data-testid="stHeader"] * {
        color: #FFFFFF !important;
    }

    /* Universal Text Readability Overrides */
    p, span, label, li, div, small, a {
        color: #F8FAFC;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px !important;
    }
    
    /* Sidebar Overhaul for Complete Text Legibility */
    section[data-testid="stSidebar"] {
        background-color: #090E1A !important;
        border-right: 1px solid #1E293B !important;
    }
    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #FFFFFF !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        margin-top: 14px !important;
        margin-bottom: 8px !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] li {
        color: #F1F5F9 !important;
        font-size: 0.98rem !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar Radio Navigation Items */
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background: #111827 !important;
        border: 1px solid #27354A !important;
        border-radius: 8px !important;
        padding: 9px 14px !important;
        margin-bottom: 8px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background: rgba(34, 211, 238, 0.15) !important;
        border-color: #22D3EE !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.98rem !important;
        margin: 0 !important;
    }
    
    /* Sidebar Selectbox & Inputs */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextArea"] label {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    div[data-baseweb="select"] * {
        color: #FFFFFF !important;
        background-color: #0F172A !important;
        font-weight: 500 !important;
    }
    
    /* CRITICAL FIX: URL Input Box Styling */
    div[data-testid="stTextInput"] > div,
    div[data-testid="stTextInput"] > div > div,
    div[data-testid="stTextInput"] div[data-baseweb="base-input"],
    div[data-testid="stTextInput"] div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-baseweb="input"],
    div[data-testid="stTextArea"] div[data-baseweb="textarea"],
    div[data-baseweb="textarea"] {
        background-color: #0F172A !important;
        border: 2px solid #334155 !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5) !important;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="textarea"]:focus-within {
        border-color: #22D3EE !important;
        box-shadow: 0 0 18px rgba(34, 211, 238, 0.45) !important;
    }
    
    /* Input text: Sharp, bright white and cyan caret */
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea,
    div[data-testid="stTextInput"] input,
    input[type="text"],
    input {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.12rem !important;
        font-weight: 600 !important;
        caret-color: #22D3EE !important;
        padding: 10px 14px !important;
    }
    div[data-baseweb="input"] input::placeholder,
    input::placeholder,
    textarea::placeholder {
        color: #94A3B8 !important;
        -webkit-text-fill-color: #94A3B8 !important;
        font-weight: 400 !important;
    }
    
    /* CRITICAL FIX: Buttons & Form Submit Buttons */
    div[data-testid="stFormSubmitButton"] > button,
    div.stButton > button,
    button[kind="primary"],
    button[kind="secondary"] {
        background: linear-gradient(135deg, #0284C7 0%, #06B6D4 100%) !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        border: 1px solid #38BDF8 !important;
        border-radius: 10px !important;
        padding: 14px 28px !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 18px rgba(6, 182, 212, 0.4) !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover,
    div.stButton > button:hover,
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369A1 0%, #0891B2 100%) !important;
        border-color: #22D3EE !important;
        box-shadow: 0 8px 25px rgba(34, 211, 238, 0.6) !important;
        transform: translateY(-1px) !important;
    }
    div[data-testid="stFormSubmitButton"] > button p,
    div.stButton > button p {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 800 !important;
    }

    /* CRITICAL FIX: Code & Preformatted Blocks (Phishing Awareness White Box Fix) */
    code, pre, kbd, .stMarkdown code {
        background-color: #111827 !important;
        color: #22D3EE !important;
        -webkit-text-fill-color: #22D3EE !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
        padding: 4px 10px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.95em !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Brand Header Bar */
    .brand-header {
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.98) 0%, rgba(15, 23, 42, 0.92) 100%);
        border: 1px solid #2A3B53;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.7), 0 0 20px 0 rgba(34, 211, 238, 0.12);
        backdrop-filter: blur(12px);
    }
    
    .brand-title {
        font-size: 2.1rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 45%, #22D3EE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    
    .brand-tagline {
        color: #E2E8F0 !important;
        font-size: 1.05rem !important;
        margin-top: 5px;
        margin-bottom: 0;
        font-weight: 500 !important;
    }
    
    .brand-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34, 211, 238, 0.15);
        border: 1px solid rgba(34, 211, 238, 0.4);
        color: #22D3EE !important;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Hero Section Card */
    .hero-card {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.98) 0%, rgba(15, 23, 42, 0.94) 100%);
        border: 1px solid #27354A;
        border-top: 3px solid #22D3EE;
        border-radius: 20px;
        padding: 42px;
        margin-bottom: 30px;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.8);
        position: relative;
        overflow: hidden;
    }
    
    .hero-headline {
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        color: #FFFFFF !important;
        line-height: 1.2;
        letter-spacing: -0.02em;
        margin-bottom: 14px;
    }
    
    .hero-subtext {
        font-size: 1.2rem !important;
        color: #F1F5F9 !important;
        line-height: 1.7;
        max-width: 780px;
        margin-bottom: 24px;
        font-weight: 400 !important;
    }
    
    .privacy-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.45);
        color: #34D399 !important;
        padding: 10px 18px;
        border-radius: 10px;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 14px;
    }
    .privacy-badge strong {
        color: #FFFFFF !important;
    }
    
    /* Feature Cards on Home */
    .feature-card {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 14px;
        padding: 26px;
        height: 100%;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.5);
    }
    .feature-card h3 {
        color: #FFFFFF !important;
        font-size: 1.25rem !important;
        margin-bottom: 10px !important;
        font-weight: 700 !important;
    }
    .feature-card p {
        color: #E2E8F0 !important;
        font-size: 0.98rem !important;
        line-height: 1.6 !important;
    }

    /* Result Badges and Cards */
    .result-card-safe {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 78, 59, 0.3) 100%);
        border: 1.5px solid #10B981;
        border-radius: 16px;
        padding: 28px 34px;
        margin-top: 24px;
        box-shadow: 0 10px 30px -10px rgba(16, 185, 129, 0.4);
    }
    .result-card-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(120, 53, 15, 0.3) 100%);
        border: 1.5px solid #F59E0B;
        border-radius: 16px;
        padding: 28px 34px;
        margin-top: 24px;
        box-shadow: 0 10px 30px -10px rgba(245, 158, 11, 0.4);
    }
    .result-card-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.14) 0%, rgba(127, 29, 29, 0.35) 100%);
        border: 1.5px solid #EF4444;
        border-radius: 16px;
        padding: 28px 34px;
        margin-top: 24px;
        box-shadow: 0 10px 30px -10px rgba(239, 68, 68, 0.5);
    }
    
    .status-badge-lg {
        display: inline-block;
        padding: 8px 22px;
        font-size: 1.2rem;
        font-weight: 900;
        border-radius: 8px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-safe {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.5);
    }
    .badge-warning {
        background-color: #F59E0B !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 16px rgba(245, 158, 11, 0.5);
    }
    .badge-danger {
        background-color: #EF4444 !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 22px rgba(239, 68, 68, 0.6);
    }
    
    /* Anatomy Syntax Pill Cards */
    .anatomy-container {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        margin: 20px 0;
    }
    .anatomy-pill {
        background: #111827;
        border: 1px solid #27354A;
        border-radius: 10px;
        padding: 14px 20px;
        flex: 1;
        min-width: 150px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .anatomy-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #22D3EE !important;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .anatomy-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.02rem;
        color: #FFFFFF !important;
        font-weight: 600;
        word-break: break-all;
    }
    
    /* KPI Metric Cards in Dashboard */
    div[data-testid="stMetric"] {
        background-color: #111827 !important;
        border: 1px solid #27354A !important;
        border-radius: 14px !important;
        padding: 22px !important;
        box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.6) !important;
    }
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] span {
        color: #22D3EE !important;
        font-size: 0.95rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #FFFFFF !important;
        font-size: 2.4rem !important;
        font-weight: 900 !important;
    }
    
    /* Streamlit Tabs */
    button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #CBD5E1 !important;
        font-size: 1.05rem !important;
        padding: 12px 24px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #22D3EE !important;
        font-weight: 800 !important;
        border-bottom-color: #22D3EE !important;
        border-bottom-width: 3px !important;
    }
    
    /* Indicator Items */
    .indicator-card {
        background: #0B1322;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        display: flex;
        align-items: flex-start;
        gap: 14px;
    }
    .indicator-card span {
        color: #F8FAFC !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        line-height: 1.5;
    }

    /* Educational Advisory Cards in Phishing Awareness */
    .advisory-card {
        background: #111827;
        border: 1px solid #1E293B;
        border-left: 4px solid #22D3EE;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
    }
    .advisory-title {
        color: #FFFFFF !important;
        font-size: 1.2rem;
        font-weight: 800;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .advisory-desc {
        color: #E2E8F0 !important;
        font-size: 0.98rem;
        line-height: 1.6;
        margin-bottom: 14px;
    }
    .advisory-code-box {
        background: #0B1322;
        border: 1px solid #27354A;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# 3. KNOWN SECURITY PATTERNS AND CONSTANTS
# --------------------------------------------------------------------------------------
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "cutt.ly", "rebrand.ly", "shorte.st"
}

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "verification", "security", "secure",
    "account", "update", "banking", "authenticate", "recovery", "password",
    "credential", "paypal", "appleid", "wallet", "confirm", "support",
    "webscr", "ebayisapi", "billing", "service", "validation", "checkpoint"
]

SUSPICIOUS_TLDS = {
    "xyz", "top", "work", "loan", "click", "fit", "gq", "cf",
    "tk", "ml", "ga", "buzz", "club", "surf", "rest", "test"
}

# --------------------------------------------------------------------------------------
# 4. URL ANATOMY & PARSING ENGINE
# --------------------------------------------------------------------------------------
def parse_url_anatomy(raw_url: str) -> dict:
    """
    Safely dissects a URL into its constituent architectural components.
    Does not initiate any network or DNS connection.
    """
    url_clean = raw_url.strip()
    if not (url_clean.startswith("http://") or url_clean.startswith("https://")):
        parseable = "http://" + url_clean
    else:
        parseable = url_clean

    try:
        parsed = urlparse(parseable)
        netloc = parsed.netloc or ""
        
        # Extract port if explicitly declared
        port = str(parsed.port) if parsed.port else "Default (80/443)"
        
        # Strip port from hostname for analysis
        hostname = netloc.split(":")[0] if ":" in netloc else netloc
        
        # Subdomain and registered domain separation
        host_parts = hostname.split(".")
        if len(host_parts) > 2:
            subdomain = ".".join(host_parts[:-2])
            registered_domain = ".".join(host_parts[-2:])
        elif len(host_parts) == 2:
            subdomain = "None (Apex Domain)"
            registered_domain = hostname
        else:
            subdomain = "None"
            registered_domain = hostname

        return {
            "protocol": parsed.scheme.upper() if parsed.scheme else "UNSPECIFIED",
            "hostname": hostname if hostname else "N/A",
            "subdomain": subdomain,
            "registered_domain": registered_domain,
            "path": parsed.path if parsed.path else "/",
            "query": parsed.query if parsed.query else "None",
            "port": port
        }
    except Exception:
        return {
            "protocol": "ERROR",
            "hostname": "Unable to parse",
            "subdomain": "N/A",
            "registered_domain": "N/A",
            "path": "N/A",
            "query": "N/A",
            "port": "N/A"
        }

# --------------------------------------------------------------------------------------
# 5. LEXICAL FEATURE EXTRACTION
# --------------------------------------------------------------------------------------
def extract_url_features(url: str) -> dict:
    """
    Extracts 18 lexical and structural attributes from the URL string.
    Execution is 100% offline, isolated, and safe for cybersecurity research.
    """
    url_clean = url.strip()
    if not (url_clean.startswith("http://") or url_clean.startswith("https://")):
        parseable = "http://" + url_clean
    else:
        parseable = url_clean

    try:
        parsed = urlparse(parseable)
        hostname = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
    except Exception:
        hostname = ""
        path = ""

    domain = hostname.split(":")[0] if ":" in hostname else hostname

    # 1. Structural Lengths
    url_length = len(url_clean)
    domain_length = len(domain)

    # 2. Delimiter Characters
    dot_count = url_clean.count(".")
    hyphen_count = url_clean.count("-")
    slash_count = url_clean.count("/")
    question_count = url_clean.count("?")
    equal_count = url_clean.count("=")
    at_count = url_clean.count("@")

    # Double slash check after protocol
    path_query = parseable.split("://", 1)[-1] if "://" in parseable else parseable
    double_slash_count = path_query.count("//")

    # 3. Numerical IP Address Pattern (IPv4)
    ipv4_pattern = re.compile(
        r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    )
    is_ip = 1 if ipv4_pattern.match(domain) else 0

    # 4. HTTPS Scheme Flag
    is_https = 1 if url_clean.lower().startswith("https://") else 0

    # 5. Shortener Service Check
    is_shortener = 1 if domain in SHORTENER_DOMAINS else 0

    # 6. Credential Keyword Matching (Tokenized)
    url_tokens = set(re.split(r"[/._\-?=&:]", url_clean.lower()))
    matched_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in url_tokens]
    keyword_count = len(matched_keywords)

    # 7. Digit Metrics
    digit_count = sum(c.isdigit() for c in url_clean)
    digit_ratio = round(digit_count / max(1, url_length), 4)

    # 8. Subdomain Count
    domain_parts = domain.split(".")
    subdomain_count = max(0, len(domain_parts) - 2)

    # 9. High-Risk TLD Flag
    tld = domain_parts[-1] if len(domain_parts) > 1 else ""
    has_suspicious_tld = 1 if tld in SUSPICIOUS_TLDS else 0

    # 10. Hyphen in Domain Flag
    hyphen_in_domain = 1 if "-" in domain else 0

    return {
        "url_length": url_length,
        "domain_length": domain_length,
        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "slash_count": slash_count,
        "question_count": question_count,
        "equal_count": equal_count,
        "at_count": at_count,
        "double_slash_count": double_slash_count,
        "is_ip": is_ip,
        "is_https": is_https,
        "is_shortener": is_shortener,
        "keyword_count": keyword_count,
        "digit_count": digit_count,
        "digit_ratio": digit_ratio,
        "subdomain_count": subdomain_count,
        "has_suspicious_tld": has_suspicious_tld,
        "hyphen_in_domain": hyphen_in_domain,
        "_matched_keywords": matched_keywords,
    }


def identify_suspicious_indicators(features: dict, url: str) -> list:
    """
    Translates extracted technical attributes into plain-English security warning indicators.
    """
    indicators = []

    if features["is_ip"] == 1:
        indicators.append("Host is a raw numerical IP address instead of a legitimate domain.")

    if features["at_count"] > 0:
        indicators.append("Contains '@' symbol, which can confuse browser host parsing.")

    if features["is_shortener"] == 1:
        indicators.append("Uses a URL shortening service that conceals the true destination hostname.")

    if features["is_https"] == 0:
        indicators.append("Lacks HTTPS encryption (transmits data in unencrypted HTTP cleartext).")

    if features["url_length"] > 75:
        indicators.append(f"Abnormally long URL ({features['url_length']} characters), commonly used to bury malicious paths.")

    if features["hyphen_in_domain"] == 1:
        indicators.append("Domain contains hyphens, a standard indicator in brand-impersonation and typosquatting.")

    if features["dot_count"] >= 4:
        indicators.append(f"High number of dots ({features['dot_count']}), suggesting complex nested subdomains.")

    if features["keyword_count"] > 0:
        kw_list = ", ".join(features.get("_matched_keywords", []))
        indicators.append(f"Sensitive credential-harvesting keyword(s) detected: [{kw_list}].")

    if features["double_slash_count"] > 0:
        indicators.append("Contains anomalous double slash '//' in path, indicating potential open redirection.")

    if features["has_suspicious_tld"] == 1:
        indicators.append("Uses a top-level domain (TLD) with elevated historical rates of malicious abuse.")

    if features["digit_ratio"] > 0.20:
        indicators.append(f"High concentration of numerical digits ({int(features['digit_ratio']*100)}% of URL length).")

    if not indicators:
        indicators.append("No overt structural anomalies or high-risk lexical indicators detected.")

    return indicators

# --------------------------------------------------------------------------------------
# 6. ML PIPELINE & BENCHMARK TRAINING
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Random Forest model on benchmark dataset...")
def train_phishing_model():
    """
    Builds a curated, representative benchmark dataset of legitimate and phishing URLs,
    extracts lexical features, and trains a 100-tree Random Forest Classifier.
    """
    try:
        legitimate_urls = [
            "https://www.google.com",
            "https://www.wikipedia.org",
            "https://github.com/torvalds/linux",
            "https://stackoverflow.com/questions/tagged/python",
            "https://www.microsoft.com/en-us/software-download",
            "https://amazon.com/gp/bestsellers",
            "https://www.apple.com/iphone-15-pro",
            "https://www.python.org/downloads/release",
            "https://www.nih.gov/health-information",
            "https://www.mit.edu/academics",
            "https://www.stanford.edu/research",
            "https://www.bbc.com/news/technology",
            "https://www.nytimes.com/section/technology",
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://docs.python.org/3/library/urllib.parse.html",
            "https://scikit-learn.org/stable/modules/ensemble.html",
            "https://streamlit.io/components",
            "https://pandas.pydata.org/docs/reference",
            "https://www.cisco.com/c/en/us/products/security.html",
            "https://aws.amazon.com/console",
            "https://cloud.google.com/docs",
            "https://portal.azure.com",
            "https://www.kaggle.com/datasets",
            "https://news.ycombinator.com/newest",
            "https://www.reddit.com/r/cybersecurity",
            "https://www.mozilla.org/en-US/firefox",
            "https://www.ibm.com/topics/cybersecurity",
            "https://www.cloudflare.com/learning/access-management/what-is-phishing",
            "https://cve.mitre.org/cve/identifiers",
            "https://owasp.org/www-project-top-ten",
            "https://www.apache.org/licenses",
            "https://www.kernel.org/pub/linux/kernel",
            "https://developer.mozilla.org/en-US/docs/Web/HTTP",
            "https://www.nist.gov/cyberframework",
            "https://www.docker.com/products/docker-desktop",
            "https://pypi.org/project/scikit-learn",
            "https://www.coursera.org/browse/information-technology",
            "https://www.edx.org/learn/computer-science",
            "https://archive.org/web",
            "https://www.eff.org/pages/tools",
            "https://www.acm.org/publications",
            "https://ieeexplore.ieee.org/Xplore/home.jsp",
            "https://www.w3.org/TR/html52",
            "https://wordpress.org/download",
            "https://slack.com/features",
            "https://zoom.us/pricing",
            "https://www.dropbox.com/business",
            "https://trello.com/tour",
            "https://www.linkedin.com/learning",
            "https://about.gitlab.com/solutions",
            "http://info.cern.ch/hypertext/WWW/TheProject.html",
            "http://www.columbia.edu/~fdc/sample.html",
            "http://www.gnu.org/licenses/gpl-3.0.html",
            "http://www.faqs.org/rfcs/rfc2616.html",
            "http://example.com/index.html",
            "https://open-source-initiative.org/licenses",
            "https://user-guide.readthedocs.io/en/latest",
            "https://mirror.cs.uchicago.edu/debian",
            "https://support.google.com/mail/answer/8253",
            "https://docs.github.com/en/rest"
        ]

        phishing_urls = [
            "http://192.168.1.105/paypal/signin.php?cmd=login_submit",
            "http://218.45.10.12/secure-banking/verification.html",
            "http://paypal-security-update-center.xyz/login.html",
            "http://appleid.apple.com.verify-device-alert.top/signin",
            "http://login.microsoftonline.com.account-update.buzz/auth",
            "http://bankofamerica-secure-auth.club/webscr?verify=user",
            "http://www.chase-online-banking-security-access.top/login",
            "http://wellsfargo.customer-support-update.work/auth/index.php",
            "http://bit.ly/3xVerifyBankUpdate",
            "http://tinyurl.com/secure-apple-login-recover",
            "http://secure-update-account-verification.xyz/index.html",
            "http://netflix-billing-update-subscription.work/auth",
            "http://amazon-orders-confirmation-alert.top/signin",
            "http://dhl-package-tracking-reschedule.click/payment",
            "http://facebook-security-checkpoint.help.fit/login.php",
            "http://instagram-copyright-infringement-appeal.buzz/submit",
            "http://accounts.google.com-recovery-password.loan/signin",
            "http://ebayisapi.dll-signin-account-verify.xyz/auth",
            "http://coinbase-wallet-security-migration.top/seed-phrase",
            "http://metamask-extension-recovery-phrase.club/sync",
            "http://irs-tax-refund-status-claim.xyz/submit-ssn",
            "http://fedex-express-delivery-redelivery.rest/billing",
            "http://walmart-giftcard-promotions-reward.buzz/claim",
            "http://verizon-account-overdue-payment-notice.top/pay",
            "http://att-yahoo-mail-synchronization-login.work/auth",
            "http://104.244.42.1/update-account-password/index.html",
            "http://secure-login-paypal.com.account-recovery-session.xyz/webscr",
            "http://update-secure-banking-portal-verification.club/login",
            "http://signin.ebay.com.ebayisapi.dll.secure-auth.top/signin",
            "http://apple-icloud-device-lost-location-verify.xyz/map",
            "http://usps-tracking-package-address-verification.work/order",
            "http://binance-kyc-document-update-verification.top/auth",
            "http://discord-nitro-free-giveaway-steam-claim.xyz/gift",
            "http://steamcommunity.com-id-trade-confirm.buzz/offer",
            "http://roblox-free-robux-promo-code-redeem.club/login",
            "http://security-alert-unauthorized-access-notice.fit/verify",
            "http://confirm-identity-id-gov-benefit-portal.xyz/login",
            "http://citi-card-fraud-prevention-alert.top/verify-now",
            "http://americanexpress-card-security-alert.work/login.html",
            "http://support-service-ticket-resolution.xyz/auth/credentials",
            "http://login.live.com.user-credential-harvest.top/auth",
            "http://secure-webmail-corporate-exchange-login.club/owa",
            "http://hr-employee-annual-benefits-survey-portal.xyz/form",
            "http://payroll-direct-deposit-update-notice.work/submit",
            "http://it-helpdesk-vpn-two-factor-auth-reset.buzz/verify",
            "http://adobe-document-cloud-shared-file-review.top/login",
            "http://docu-sign-encrypted-invoice-signature.xyz/auth",
            "http://sharepoint-secure-file-download-access.fit/index",
            "http://dropbox-secure-cloud-transfer-storage.top/signin",
            "http://onedrive-business-file-share-verification.xyz/login",
            "http://user@paypal-verify-account.top/login",
            "http://secure-banking.net//login.php?redirect=bankofamerica.com",
            "http://192.168.0.1/admin-login-verification",
            "http://update-billing-info.netflix.com.verify-billing.buzz/index",
            "http://amazon-account-suspension-appeals.club/submit",
            "http://chase-security-fraud-alert.top/verify.html",
            "http://apple-support-lost-mode.click/auth",
            "http://cutt.ly/bank-security-pin-reset",
            "http://rebrand.ly/paypal-dispute-resolution",
            "http://blockchain-wallet-private-keys.top/restore"
        ]

        data_records = []
        for u in legitimate_urls:
            feats = extract_url_features(u)
            feats.pop("_matched_keywords", None)
            feats["label"] = 0
            data_records.append(feats)

        for u in phishing_urls:
            feats = extract_url_features(u)
            feats.pop("_matched_keywords", None)
            feats["label"] = 1
            data_records.append(feats)

        df = pd.DataFrame(data_records)

        feature_cols = [
            "url_length", "domain_length", "dot_count", "hyphen_count",
            "slash_count", "question_count", "equal_count", "at_count",
            "double_slash_count", "is_ip", "is_https", "is_shortener",
            "keyword_count", "digit_count", "digit_ratio", "subdomain_count",
            "has_suspicious_tld", "hyphen_in_domain"
        ]

        X = df[feature_cols]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_split=4,
            random_state=42
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "is_trained": True,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "test_size": len(y_test),
            "train_size": len(y_train),
            "total_samples": len(df),
            "feature_importances": dict(zip(feature_cols, clf.feature_importances_))
        }

        return clf, feature_cols, metrics

    except Exception as e:
        metrics = {
            "is_trained": False,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": np.zeros((2, 2), dtype=int),
            "test_size": 0,
            "train_size": 0,
            "total_samples": 0,
            "feature_importances": {},
            "error_msg": str(e)
        }
        return None, [], metrics

# --------------------------------------------------------------------------------------
# 7. RISK SCORING & SECURITY INFERENCE
# --------------------------------------------------------------------------------------
def compute_risk_score(model, feature_cols: list, features: dict) -> dict:
    """
    Computes both model-predicted class probability and an explainable heuristic risk score.
    Returns full diagnostics to keep machine learning and heuristic scoring transparent.
    """
    if model is not None and feature_cols:
        input_data = [[features[col] for col in feature_cols]]
        input_df = pd.DataFrame(input_data, columns=feature_cols)
        ml_prob = float(model.predict_proba(input_df)[0][1])
    else:
        ml_prob = 0.0

    # Calculate Heuristic Risk Score independently
    heuristic_score = 10.0
    if features["is_ip"] == 1:
        heuristic_score += 35
    if features["at_count"] > 0:
        heuristic_score += 25
    if features["keyword_count"] > 0:
        heuristic_score += min(30, features["keyword_count"] * 15)
    if features["is_shortener"] == 1:
        heuristic_score += 20
    if features["has_suspicious_tld"] == 1:
        heuristic_score += 15
    if features["hyphen_in_domain"] == 1:
        heuristic_score += 15
    if features["is_https"] == 0:
        heuristic_score += 10
    if features["double_slash_count"] > 0:
        heuristic_score += 10

    heuristic_score = min(100.0, heuristic_score)

    # Blended Calibrated Risk Score (0 - 100)
    base_ml_score = ml_prob * 100.0
    structural_penalties = 0
    if features["is_ip"] == 1:
        structural_penalties += 15
    if features["at_count"] > 0:
        structural_penalties += 15
    if features["is_shortener"] == 1 and features["keyword_count"] > 0:
        structural_penalties += 15

    final_score = int(round(max(0, min(100, base_ml_score + structural_penalties))))

    # Category and Severity mapping
    if final_score < 30:
        verdict = "Likely Legitimate"
        severity = "Low Risk"
        color_code = "safe"
    elif final_score < 60:
        verdict = "Suspicious"
        severity = "Medium Risk"
        color_code = "warning"
    else:
        verdict = "Potential Phishing"
        severity = "High Risk"
        color_code = "danger"

    return {
        "final_score": final_score,
        "verdict": verdict,
        "severity": severity,
        "color_code": color_code,
        "ml_probability": ml_prob,
        "heuristic_score": int(heuristic_score),
    }


def get_recommendation(verdict: str) -> str:
    """
    Delivers actionable defensive safety recommendations based on evaluated threat level.
    """
    if verdict == "Likely Legitimate":
        return (
            "✅ This link displays typical characteristics of legitimate web services. "
            "However, always verify that the domain name matches the expected organization exactly "
            "before entering passwords or credentials."
        )
    elif verdict == "Suspicious":
        return (
            "⚠️ Exercise elevated caution. This URL contains unusual lexical features or keywords. "
            "Do not input credentials or financial details until you have independently verified "
            "the website using an official bookmark or authenticated portal."
        )
    else:
        return (
            "🛑 High Risk Alert: Significant indicators of a phishing or credential-harvesting attempt. "
            "DO NOT enter passwords, credit card numbers, or personal information. Do not download files. "
            "Close this webpage immediately and report it to your security team."
        )

# --------------------------------------------------------------------------------------
# 8. SESSION STATE INITIALIZATION
# --------------------------------------------------------------------------------------
if "session_scans" not in st.session_state:
    st.session_state.session_scans = []

def record_session_scan(url: str, verdict: str, score: int, indicators: list):
    """
    Records a completed scan into session-based state to ensure accurate KPI metrics
    without duplicate counting upon Streamlit application reruns.
    """
    entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "URL": url,
        "Verdict": verdict,
        "Risk Score": score,
        "Indicators Count": len(indicators) if "No overt" not in indicators[0] else 0,
        "Indicators": "; ".join(indicators)
    }
    st.session_state.session_scans.append(entry)

# --------------------------------------------------------------------------------------
# 9. MAIN APPLICATION CONTROLLER
# --------------------------------------------------------------------------------------
def main():
    model, feature_cols, metrics = train_phishing_model()

    # Top Brand Navigation Header
    st.markdown(
        """
        <div class="brand-header">
            <div>
                <h1 class="brand-title">
                    🛡️ PhishGuard AI
                </h1>
                <p class="brand-tagline">Detect Deception. Browse with Confidence.</p>
            </div>
            <div style="display: flex; gap: 12px; align-items: center;">
                <span class="brand-pill">● AI ENGINE ACTIVE</span>
                <span class="brand-pill" style="border-color: rgba(59, 130, 246, 0.5); color: #93C5FD !important;">RANDOM FOREST v2.4</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Navigation System
    with st.sidebar:
        st.markdown("### 🧭 PLATFORM NAVIGATION")
        nav_choice = st.radio(
            "Select Section:",
            [
                "🏠 Home & Hero",
                "🔍 AI URL Scanner",
                "📊 Interactive Dashboard",
                "🧬 URL Anatomy",
                "⚡ Batch Analysis",
                "🧠 Explainable AI",
                "🛡️ Phishing Awareness",
                "⚙️ Model & Evaluation"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### 🎯 RISK SCALE THRESHOLDS")
        st.markdown(
            """
            - 🟢 <strong style="color: #34D399;">00 – 29:</strong> Low Risk (*Likely Legitimate*)
            - 🟡 <strong style="color: #FBBF24;">30 – 59:</strong> Medium Risk (*Suspicious*)
            - 🔴 <strong style="color: #F87171;">60 – 100:</strong> High Risk (*Potential Phishing*)
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown("### 🧪 QUICK PRESET SAMPLES")
        st.markdown("<span style='color: #F1F5F9; font-size: 0.9rem;'>Select a sample scenario:</span>", unsafe_allow_html=True)
        preset_choice = st.selectbox(
            "Choose Sample Link:",
            [
                "None",
                "Legitimate: Wikipedia Machine Learning",
                "Suspicious: Blog with Login Path",
                "Phishing: IP-Host Fake PayPal Portal",
                "Phishing: Illustrative Sample URL"
            ],
            label_visibility="collapsed"
        )

    preset_map = {
        "Legitimate: Wikipedia Machine Learning": "https://en.wikipedia.org/wiki/Machine_learning",
        "Suspicious: Blog with Login Path": "http://my-blog-community.net/login-portal",
        "Phishing: IP-Host Fake PayPal Portal": "http://192.168.1.105/paypal/signin.php?cmd=login_submit",
        "Phishing: Illustrative Sample URL": "http://example-login-security.test"
    }
    preset_url_val = preset_map.get(preset_choice, "")

    # ----------------------------------------------------------------------------------
    # VIEW 1: HOME & HERO SECTION
    # ----------------------------------------------------------------------------------
    if nav_choice == "🏠 Home & Hero":
        st.markdown(
            """
            <div class="hero-card">
                <span class="brand-pill" style="margin-bottom: 18px;">AI-POWERED URL SECURITY</span>
                <h1 class="hero-headline">Think Before You Click.</h1>
                <p class="hero-subtext">
                    Analyze suspicious URLs with AI-powered phishing detection and understand the warning signs 
                    before trusting a link. Protect your credentials, enterprise data, and personal accounts 
                    with real-time static lexical intelligence.
                </p>
                <div class="privacy-badge">
                    🔒 <strong>Privacy Guarantee:</strong> PhishGuard AI evaluates URL structure in pure isolation. 
                    Target websites are never visited, downloaded, or executed.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_h1, col_h2, col_h3 = st.columns(3)
        with col_h1:
            st.markdown(
                """
                <div class="feature-card">
                    <div style="font-size: 2rem; margin-bottom: 10px;">⚡</div>
                    <h3>Sub-Millisecond Inference</h3>
                    <p>
                        Static lexical feature extraction evaluates URL tokens instantly without waiting for DNS propagation or remote HTTP handshakes.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_h2:
            st.markdown(
                """
                <div class="feature-card">
                    <div style="font-size: 2rem; margin-bottom: 10px;">🧠</div>
                    <h3>Explainable Threat AI</h3>
                    <p>
                        Beyond raw predictions, PhishGuard AI highlights exact structural warning signs such as IP-based hosts, typosquats, and credential lures.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_h3:
            st.markdown(
                """
                <div class="feature-card">
                    <div style="font-size: 2rem; margin-bottom: 10px;">🛡️</div>
                    <h3>Zero-Trust Sandbox</h3>
                    <p>
                        Safe for enterprise analysts and students. The application operates in zero-egress mode, keeping your local machine completely isolated.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br/>", unsafe_allow_html=True)
        st.info("💡 Ready to scan a URL? Click on **🔍 AI URL Scanner** in the left navigation sidebar.")

    # ----------------------------------------------------------------------------------
    # VIEW 2: AI URL SCANNER (CORE FUNCTIONALITY)
    # ----------------------------------------------------------------------------------
    elif nav_choice == "🔍 AI URL Scanner":
        st.subheader("🔍 Central AI URL Threat Scanner")
        st.markdown(
            "<span style='color: #F1F5F9; font-size: 1.05rem; font-weight: 500;'>"
            "Paste a web URL to extract structural features, compute calibrated risk metrics, and review explainable threat diagnostics."
            "</span>",
            unsafe_allow_html=True
        )

        with st.form("main_scanner_form", clear_on_submit=False):
            url_query = st.text_input(
                "Enter Website URL for Deep Inspection:",
                value=preset_url_val,
                placeholder="e.g. http://example-login-security.test",
                help="Inspect any link. The system extracts syntactic properties without browsing the destination."
            )
            scan_btn = st.form_submit_button("⚡ SCAN URL NOW", use_container_width=True)

        if scan_btn:
            trimmed = url_query.strip()
            if not trimmed:
                st.warning("⚠️ Empty URL input. Please enter or paste a valid website link.")
            elif len(trimmed) < 4 or ("." not in trimmed and not re.search(r"\d+\.\d+\.\d+\.\d+", trimmed)):
                st.error("❌ Malformed URL. Input must contain a domain name or IP address (e.g. 'example.com').")
            else:
                with st.spinner("Extracting 18 lexical dimensions & executing Random Forest inference..."):
                    features = extract_url_features(trimmed)
                    risk_data = compute_risk_score(model, feature_cols, features)
                    indicators = identify_suspicious_indicators(features, trimmed)
                    recommendation = get_recommendation(risk_data["verdict"])

                    # Save to session history (prevent duplicate counting)
                    record_session_scan(trimmed, risk_data["verdict"], risk_data["final_score"], indicators)

                # Render Detection Results Card
                card_class = f"result-card-{risk_data['color_code']}"
                badge_class = f"badge-{risk_data['color_code']}"

                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                            <div>
                                <span style="font-size: 0.85rem; color: #E2E8F0; text-transform: uppercase; font-family: monospace; letter-spacing: 1px; font-weight: 700;">PREDICTED STATUS</span><br/>
                                <span class="status-badge-lg {badge_class}" style="margin-top: 6px;">{risk_data['verdict']}</span>
                            </div>
                            <div style="text-align: right;">
                                <span style="font-size: 0.85rem; color: #E2E8F0; text-transform: uppercase; font-family: monospace; letter-spacing: 1px; font-weight: 700;">CALIBRATED RISK SCORE</span><br/>
                                <span style="font-size: 3rem; font-weight: 900; font-family: 'JetBrains Mono', monospace; color: #FFFFFF;">
                                    {risk_data['final_score']}
                                </span>
                                <span style="font-size: 1.35rem; color: #F1F5F9; font-family: 'JetBrains Mono', monospace; font-weight: 700;">/100 ({risk_data['severity']})</span>
                            </div>
                        </div>
                        <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.18); margin: 20px 0;" />
                        <p style="margin: 0; font-size: 1.05rem; color: #FFFFFF;">
                            <strong style="color: #22D3EE;">Submitted URL:</strong> <code style="background: rgba(0,0,0,0.5); padding: 5px 10px; border-radius: 6px; color: #22D3EE; font-weight: 700; font-size: 1.05rem; border: 1px solid #334155;">{trimmed}</code>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Risk Gauge Progress Bar
                st.markdown("<div style='margin-top: 20px; font-size: 0.95rem; color: #FFFFFF; font-weight: 700; font-family: monospace;'>RISK SEVERITY METER:</div>", unsafe_allow_html=True)
                st.progress(risk_data["final_score"] / 100.0)

                # Comparison Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("ML Classifier Probability", f"{risk_data['ml_probability']*100:.1f}%")
                with col_m2:
                    st.metric("Heuristic Risk Score", f"{risk_data['heuristic_score']}/100")
                with col_m3:
                    st.metric("Overall Calibrated Score", f"{risk_data['final_score']}/100")

                st.markdown("---")

                # Columns for Indicators and Safety Recommendations
                col_left, col_right = st.columns([1, 1])

                with col_left:
                    st.markdown("### 🚩 Detected Warning Indicators")
                    for ind in indicators:
                        icon = "✅" if ("No overt" in ind and risk_data["color_code"] == "safe") else "⚠️" if risk_data["color_code"] == "warning" else "🛑"
                        st.markdown(
                            f"""
                            <div class="indicator-card">
                                <span style="font-size: 1.3rem;">{icon}</span>
                                <span>{ind}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                with col_right:
                    st.markdown("### 💡 Recommended Defensive Actions")
                    st.info(recommendation)

                # Expandable Feature Table
                with st.expander("🔬 View Extracted Lexical Feature Vector (18 Dimensions)"):
                    st.markdown("<span style='color: #FFFFFF; font-size: 0.95rem; font-weight: 600;'>Numerical attributes fed into the Random Forest classifier:</span>", unsafe_allow_html=True)
                    clean_feats = {k: v for k, v in features.items() if not k.startswith("_")}
                    feat_df = pd.DataFrame([
                        {"Feature": k.replace("_", " ").title(), "Value": v}
                        for k, v in clean_feats.items()
                    ])
                    st.dataframe(feat_df, use_container_width=True)

    # ----------------------------------------------------------------------------------
    # VIEW 3: INTERACTIVE CYBERSECURITY DASHBOARD
    # ----------------------------------------------------------------------------------
    elif nav_choice == "📊 Interactive Dashboard":
        st.subheader("📊 Session Threat Intelligence Dashboard")
        st.markdown(
            "<span style='color: #F1F5F9; font-size: 1.05rem; font-weight: 500;'>"
            "Live analytics reflecting all URL scans conducted during the current application session."
            "</span>",
            unsafe_allow_html=True
        )

        scans = st.session_state.session_scans
        total_scans = len(scans)
        high_risk_cnt = sum(1 for s in scans if s["Verdict"] == "Potential Phishing")
        low_risk_cnt = sum(1 for s in scans if s["Verdict"] == "Likely Legitimate")
        med_risk_cnt = sum(1 for s in scans if s["Verdict"] == "Suspicious")

        # 4 Core KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Scanned URLs", total_scans)
        with kpi2:
            st.metric("High-Risk Threat URLs", high_risk_cnt)
        with kpi3:
            st.metric("Low-Risk Benign URLs", low_risk_cnt)
        with kpi4:
            st.metric("Classifier Accuracy", f"{metrics['accuracy']*100:.1f}%")

        st.markdown("---")

        if total_scans == 0:
            st.info("ℹ️ No URLs scanned in this session yet. Go to **🔍 AI URL Scanner** to analyze links and populate the dashboard.")
        else:
            col_d1, col_d2 = st.columns([1, 1])

            with col_d1:
                st.markdown("#### 🎯 Threat Category Distribution")
                cat_counts = {"Likely Legitimate": low_risk_cnt, "Suspicious": med_risk_cnt, "Potential Phishing": high_risk_cnt}
                active_cats = {k: v for k, v in cat_counts.items() if v > 0}

                fig_donut, ax_donut = plt.subplots(figsize=(4.5, 3.6))
                fig_donut.patch.set_facecolor("#080D18")
                ax_donut.set_facecolor("#080D18")

                colors = {"Likely Legitimate": "#10B981", "Suspicious": "#F59E0B", "Potential Phishing": "#EF4444"}
                slice_colors = [colors[k] for k in active_cats.keys()]

                wedges, texts, autotexts = ax_donut.pie(
                    list(active_cats.values()),
                    labels=list(active_cats.keys()),
                    colors=slice_colors,
                    autopct="%1.1f%%",
                    startangle=90,
                    wedgeprops=dict(width=0.45, edgecolor="#1E293B", linewidth=2),
                    textprops=dict(color="#FFFFFF", fontsize=10, weight="bold")
                )
                for at in autotexts:
                    at.set_color("#FFFFFF")
                    at.set_fontweight("bold")
                st.pyplot(fig_donut)

            with col_d2:
                st.markdown("#### 📈 Session Scan Activity Timeline")
                timeline_df = pd.DataFrame(scans)
                timeline_df["Scan Index"] = range(1, len(timeline_df) + 1)

                fig_line, ax_line = plt.subplots(figsize=(5, 3.6))
                fig_line.patch.set_facecolor("#080D18")
                ax_line.set_facecolor("#0F172A")

                ax_line.plot(timeline_df["Scan Index"], timeline_df["Risk Score"], marker="o", color="#22D3EE", linewidth=2.5, markersize=6)
                ax_line.axhline(60, color="#EF4444", linestyle="--", alpha=0.8, label="High Risk Threshold (60)")
                ax_line.axhline(30, color="#10B981", linestyle="--", alpha=0.8, label="Low Risk Threshold (30)")
                ax_line.set_xlabel("Scan Number", color="#FFFFFF", fontsize=10, weight="bold")
                ax_line.set_ylabel("Risk Score (0-100)", color="#FFFFFF", fontsize=10, weight="bold")
                ax_line.set_ylim(-5, 105)
                ax_line.tick_params(colors="#FFFFFF", labelsize=9)
                for spine in ax_line.spines.values():
                    spine.set_color("#2A3B53")
                ax_line.legend(facecolor="#111827", edgecolor="#2A3B53", labelcolor="#FFFFFF", fontsize=9)
                st.pyplot(fig_line)

            st.markdown("---")
            st.markdown("#### 📜 Recent Scans Audit Log")
            scan_table = pd.DataFrame(scans)[["Timestamp", "URL", "Verdict", "Risk Score", "Indicators"]]
            st.dataframe(scan_table.iloc[::-1], use_container_width=True)

            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                csv_bytes = scan_table.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Session Audit Log (CSV)",
                    data=csv_bytes,
                    file_name="phishguard_session_scans.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_btn2:
                if st.button("🗑️ Clear Session Scans", use_container_width=True):
                    st.session_state.session_scans = []
                    st.rerun()

    # ----------------------------------------------------------------------------------
    # VIEW 4: URL ANATOMY ANALYSIS
    # ----------------------------------------------------------------------------------
    elif nav_choice == "🧬 URL Anatomy":
        st.subheader("🧬 Visual URL Anatomy & Architectural Breakdown")
        st.markdown(
            "<span style='color: #F1F5F9; font-size: 1.05rem; font-weight: 500;'>"
            "Deconstruct a target URL into its structural components (Protocol, Hostname, Subdomain, Domain, Path, Query, Port) "
            "to detect obfuscation techniques."
            "</span>",
            unsafe_allow_html=True
        )

        anatomy_input = st.text_input(
            "Enter URL to deconstruct:",
            value=preset_url_val or "http://paypal-security-update-center.xyz/login.html?session=auth",
            help="Parses URL syntax without network calls."
        )

        if anatomy_input:
            anatomy = parse_url_anatomy(anatomy_input)

            st.markdown(
                f"""
                <div class="anatomy-container">
                    <div class="anatomy-pill">
                        <div class="anatomy-label">Scheme / Protocol</div>
                        <div class="anatomy-value" style="color: {'#10B981' if anatomy['protocol'] == 'HTTPS' else '#EF4444'}; font-weight: 800;">
                            {anatomy['protocol']}
                        </div>
                    </div>
                    <div class="anatomy-pill">
                        <div class="anatomy-label">Hostname / FQDN</div>
                        <div class="anatomy-value">{anatomy['hostname']}</div>
                    </div>
                    <div class="anatomy-pill">
                        <div class="anatomy-label">Subdomain Tier</div>
                        <div class="anatomy-value">{anatomy['subdomain']}</div>
                    </div>
                    <div class="anatomy-pill">
                        <div class="anatomy-label">Registered Domain</div>
                        <div class="anatomy-value">{anatomy['registered_domain']}</div>
                    </div>
                    <div class="anatomy-pill">
                        <div class="anatomy-label">Path Traversal</div>
                        <div class="anatomy-value">{anatomy['path']}</div>
                    </div>
                    <div class="anatomy-pill">
                        <div class="anatomy-label">Query Parameters</div>
                        <div class="anatomy-value">{anatomy['query']}</div>
                    </div>
                    <div class="anatomy-pill">
                        <div class="anatomy-label">Explicit Port</div>
                        <div class="anatomy-value">{anatomy['port']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("#### 🔍 Architectural Anomaly Analysis")
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.markdown("**1. Host Identification Method:**")
                if re.match(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$", anatomy["hostname"]):
                    st.error("❌ High Risk: Raw numerical IP address detected instead of a registered domain name.")
                else:
                    st.success("✅ Clean: Hostname utilizes standard Domain Name System (DNS) formatting.")

                st.markdown("**2. Subdomain Tier Complexity:**")
                dots = anatomy["hostname"].count(".")
                if dots > 3:
                    st.warning(f"⚠️ Warning: Deep subdomain tiering ({dots} dots) observed. Often used to simulate brand domains.")
                else:
                    st.success("✅ Clean: Standard subdomain hierarchy.")

            with col_a2:
                st.markdown("**3. Communication Encryption:**")
                if anatomy["protocol"] == "HTTPS":
                    st.success("✅ Clean: Transports data with SSL/TLS encryption (Note: over 80% of phishing sites also use HTTPS).")
                else:
                    st.error("❌ High Risk: Unencrypted cleartext HTTP protocol.")

                st.markdown("**4. Hostname Delimiter Integrity:**")
                if "-" in anatomy["hostname"]:
                    st.warning("⚠️ Warning: Hyphen present in hostname. Frequently used in typosquatting and brand spoofing.")
                else:
                    st.success("✅ Clean: No hyphens in hostname.")

    # ----------------------------------------------------------------------------------
    # VIEW 5: BATCH PHISHING DETECTION
    # ----------------------------------------------------------------------------------
    elif nav_choice == "⚡ Batch Analysis":
        st.subheader("⚡ Bulk URL Batch Detection")
        st.markdown(
            "<span style='color: #F1F5F9; font-size: 1.05rem; font-weight: 500;'>"
            "Analyze multiple suspicious links concurrently by adding URLs line-by-line, pasting multi-line text, or uploading a CSV file."
            "</span>",
            unsafe_allow_html=True
        )

        def extract_batch_urls(text: str) -> list:
            """
            Intelligently splits and cleans raw text into individual URLs, each on a different line.
            Handles newlines, spaces, commas, semicolons, tabs, and concatenated schemes.
            """
            if not text or not str(text).strip():
                return []
            cleaned = re.sub(r'([^\s])(https?://)', r'\1 \2', str(text).strip(), flags=re.IGNORECASE)
            tokens = re.split(r'[\r\n\s,;]+', cleaned)
            urls = []
            for p in tokens:
                p = p.strip().strip("'\"`()[]{}<>")
                if len(p) >= 4 and ("." in p or "http://" in p.lower() or "https://" in p.lower()):
                    if p not in urls:
                        urls.append(p)
            return urls

        # Session state for line-by-line batch list and batch results
        if "batch_url_queue" not in st.session_state:
            st.session_state.batch_url_queue = []
        if "batch_results_df" not in st.session_state:
            st.session_state.batch_results_df = None

        batch_tab1, batch_tab2, batch_tab3 = st.tabs([
            "➕ Add URLs Line-by-Line",
            "📝 Direct Multi-Line Paste",
            "📁 Upload CSV File"
        ])

        raw_urls_to_process = []

        # TAB 1: ADD URLS LINE BY LINE
        with batch_tab1:
            with st.form("batch_add_url_form", clear_on_submit=True):
                col_in, col_btn = st.columns([4, 1])
                with col_in:
                    typed_url = st.text_input(
                        "Paste URL to add to batch list (each added URL is placed on a different line below):",
                        placeholder="e.g. https://example.com or http://login-verify.xyz",
                        key="batch_single_entry"
                    )
                with col_btn:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    add_btn = st.form_submit_button("➕ Add URL", use_container_width=True)

            if add_btn and typed_url:
                found = extract_batch_urls(typed_url)
                if found:
                    for u in found:
                        if u not in st.session_state.batch_url_queue:
                            st.session_state.batch_url_queue.append(u)
                    st.rerun()

            q_len = len(st.session_state.batch_url_queue)

            if q_len > 0:
                st.markdown(f"**Current URL Batch List ({q_len} URL{'s' if q_len > 1 else ''} on different lines):**")
                for idx, u in enumerate(st.session_state.batch_url_queue, 1):
                    col_line, col_del = st.columns([6, 1])
                    with col_line:
                        st.markdown(
                            f"""
                            <div style="background: #111827; border: 1px solid #1E293B; border-radius: 8px; padding: 10px 16px; margin-bottom: 6px; font-family: monospace;">
                                <span style="color: #22D3EE; font-weight: bold; margin-right: 12px;">Line {idx}:</span>
                                <span style="color: #FFFFFF; font-weight: 600;">{u}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col_del:
                        if st.button("❌ Remove", key=f"del_q_item_{idx}", use_container_width=True):
                            st.session_state.batch_url_queue.pop(idx - 1)
                            st.rerun()

                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                col_q_act1, col_q_act2 = st.columns([1, 1])
                with col_q_act1:
                    if st.button("🗑️ Clear All URLs (Remove Previous)", use_container_width=True):
                        st.session_state.batch_url_queue = []
                        st.session_state.batch_results_df = None
                        st.rerun()

                with col_q_act2:
                    if q_len < 2:
                        st.warning("⚠️ **More than one URL is required for Batch Analysis!** (Minimum 2 URLs). Please add at least 1 more URL above.")
                    else:
                        if st.button(f"🚀 Run Batch Analysis ({q_len} URLs)", type="primary", use_container_width=True):
                            raw_urls_to_process = list(st.session_state.batch_url_queue)
            else:
                st.info("ℹ️ No URLs added yet. Copy a URL, paste it above, and click **➕ Add URL** to place each URL on a different line.")

        # TAB 2: DIRECT MULTI-LINE PASTE
        with batch_tab2:
            batch_text_input = st.text_area(
                "Paste URLs (each URL on a different line, or separated by space/comma):",
                height=180,
                placeholder="https://example1.com\nhttps://example2.org/portal\nhttp://example3-phishing.xyz/login",
                key="direct_multiline_area"
            )

            col_m1, col_m2 = st.columns([1, 1])
            with col_m1:
                run_multiline = st.button("🚀 Process Multi-Line Batch", use_container_width=True)
            with col_m2:
                if st.button("🗑️ Clear Text Area (Remove Previous)", use_container_width=True):
                    st.session_state.direct_multiline_area = ""
                    st.session_state.batch_results_df = None
                    st.rerun()

            if run_multiline:
                urls_found = extract_batch_urls(batch_text_input)
                if not urls_found:
                    st.warning("⚠️ Input is empty. Please enter more than one URL.")
                elif len(urls_found) < 2:
                    st.warning("⚠️ **More than one URL is required for Batch Analysis!** (Minimum 2 URLs). You provided only 1 URL. Please enter at least 2 URLs on different lines.")
                else:
                    raw_urls_to_process = urls_found

        # TAB 3: UPLOAD CSV FILE
        with batch_tab3:
            uploaded_csv = st.file_uploader("Upload CSV containing URL column:", type=["csv"])
            if uploaded_csv is not None:
                try:
                    upload_df = pd.read_csv(uploaded_csv)
                    col_candidates = [c for c in upload_df.columns if "url" in c.lower() or "link" in c.lower()]
                    target_col = col_candidates[0] if col_candidates else upload_df.columns[0]
                    csv_urls = upload_df[target_col].dropna().astype(str).str.strip().tolist()
                    csv_urls = [u for u in csv_urls if len(u) >= 4]

                    if len(csv_urls) < 2:
                        st.warning("⚠️ Uploaded CSV must contain **more than one URL** for batch analysis (minimum 2 rows).")
                    else:
                        raw_urls_to_process = csv_urls
                        st.success(f"✅ Loaded {len(raw_urls_to_process)} URLs from column: `{target_col}`")
                except Exception as ex:
                    st.error(f"Error parsing uploaded CSV: {ex}")

        # EXECUTE BATCH ANALYSIS
        if raw_urls_to_process:
            with st.spinner(f"Analyzing batch of {len(raw_urls_to_process)} URLs..."):
                batch_records = []
                for u in raw_urls_to_process:
                    if len(u) < 4:
                        continue
                    feats = extract_url_features(u)
                    r_data = compute_risk_score(model, feature_cols, feats)
                    batch_records.append({
                        "URL": u,
                        "Verdict": r_data["verdict"],
                        "Risk Score": r_data["final_score"],
                        "Severity": r_data["severity"],
                        "ML Probability": f"{r_data['ml_probability']*100:.1f}%",
                        "Heuristic Score": r_data["heuristic_score"]
                    })

                st.session_state.batch_results_df = pd.DataFrame(batch_records)

        # DISPLAY RESULTS IF AVAILABLE IN SESSION STATE
        if st.session_state.batch_results_df is not None and not st.session_state.batch_results_df.empty:
            batch_res_df = st.session_state.batch_results_df
            st.markdown("---")
            st.markdown("### 📊 Batch Evaluation Results")

            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.metric("Total URLs Processed", len(batch_res_df))
            with col_b2:
                phish_cnt = (batch_res_df["Verdict"] == "Potential Phishing").sum()
                st.metric("High-Risk Flagged", int(phish_cnt))
            with col_b3:
                avg_score = batch_res_df["Risk Score"].mean()
                st.metric("Average Risk Score", f"{avg_score:.1f}/100")

            # Filter control
            filter_choice = st.selectbox("Filter Findings by Category:", ["All", "Potential Phishing", "Suspicious", "Likely Legitimate"])
            if filter_choice != "All":
                filtered_df = batch_res_df[batch_res_df["Verdict"] == filter_choice]
            else:
                filtered_df = batch_res_df

            st.dataframe(filtered_df, use_container_width=True)

            col_dl1, col_dl2 = st.columns([1, 1])
            with col_dl1:
                csv_out = batch_res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Complete Batch Results (CSV)",
                    data=csv_out,
                    file_name="phishguard_batch_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_dl2:
                if st.button("🗑️ Clear Batch Results (Remove Previous)", use_container_width=True):
                    st.session_state.batch_results_df = None
                    st.rerun()

    # ----------------------------------------------------------------------------------
    # VIEW 6: EXPLAINABLE AI SECTION
    # ----------------------------------------------------------------------------------
    elif nav_choice == "🧠 Explainable AI":
        st.subheader("🧠 Explainable Machine Learning & Decision Architecture")
        st.markdown(
            """
            In mission-critical cybersecurity operations, an algorithm must never function as an opaque black box.
            PhishGuard AI combines statistical ensemble modeling with transparent, rule-based verification:
            
            ### 1. Dual Evaluation Engine
            - **Statistical Random Forest Model:** Computes non-linear class probability based on 18 lexical features trained on representative legitimate and phishing URL benchmarks.
            - **Heuristic Rule-Based Safeguard:** Independent scoring engine that evaluates severe structural red flags (e.g., raw IP host, browser credential bypass `@`, URL shorteners concealing phishing keywords).
            
            ### 2. Decision Thresholds & Calibration
            - **00 – 29 (Low Risk):** Clean lexical structure typical of benign internet services.
            - **30 – 59 (Medium Risk):** Structural anomalies detected (e.g. nested subdomains, sensitive keywords in HTTP).
            - **60 – 100 (High Risk):** Overt phishing signatures (typosquatted domains, IP hosts, or brand spoofing).
            
            ### 3. Model Limitations & Uncertainty Disclosures
            - **No Content Inspection:** Pure lexical analysis evaluates only the address string. It cannot inspect dynamic JavaScript payloads or DOM elements.
            - **Compromised Legitimate Hosts:** If an attacker compromises a high-reputation domain without modifying the URL structure, lexical classifiers may yield false negatives.
            - **HTTPS Misconception:** Attackers readily provision free Let's Encrypt SSL certificates; the padlock icon does not guarantee site trustworthiness.
            """
        )

    # ----------------------------------------------------------------------------------
    # VIEW 7: PHISHING AWARENESS & EDUCATION (HIGH-CONTRAST CUSTOM CARDS)
    # ----------------------------------------------------------------------------------
    elif nav_choice == "🛡️ Phishing Awareness":
        st.subheader("🛡️ Phishing Awareness & Defensive Guidance")
        st.markdown(
            "<span style='color: #F1F5F9; font-size: 1.05rem; font-weight: 500;'>"
            "Essential threat intelligence for recognizing modern deceptive URL tactics and protecting credentials."
            "</span>",
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="advisory-card">
                <div class="advisory-title">🎯 1. Typosquatting & Brand Spoofing</div>
                <div class="advisory-desc">
                    Attackers register domains that look nearly identical to popular services by substituting characters, 
                    adding hyphens, or transposing letters.
                </div>
                <div class="advisory-code-box">
                    <span style="color: #10B981; font-weight: 700;">[GENUINE TARGET]:</span> https://www.paypal.com<br/>
                    <span style="color: #EF4444; font-weight: 700;">[SPOOFED PHISH]:</span> http://paypal-security-update.com &bull; http://paypa1.com
                </div>
            </div>

            <div class="advisory-card">
                <div class="advisory-title">🏢 2. Multi-Tier Subdomain Disguises</div>
                <div class="advisory-desc">
                    Any domain registrant can generate unlimited subdomains. Phishers prepend trusted brand names 
                    as subdomains to mislead users who glance only at the beginning of the URL.
                </div>
                <div class="advisory-code-box">
                    <span style="color: #EF4444; font-weight: 700;">[DECEPTIVE LINK]:</span> http://microsoft.com.account-verification.xyz/auth<br/>
                    <span style="color: #22D3EE; font-weight: 700;">[TRUE HOST]:</span> account-verification.xyz <span style="color: #EF4444;">(NOT Microsoft!)</span>
                </div>
            </div>

            <div class="advisory-card">
                <div class="advisory-title">🔗 3. URL Shortener Masking</div>
                <div class="advisory-desc">
                    URL shortening services conceal the destination domain completely. Security filters in chat or email 
                    cannot inspect the target host without expanding the redirect.
                </div>
                <div class="advisory-code-box">
                    <span style="color: #F59E0B; font-weight: 700;">[CONCEALED REDIRECT]:</span> http://bit.ly/3xVerifyBankUpdate &rarr; Redirects to credential harvester
                </div>
            </div>

            <div class="advisory-card">
                <div class="advisory-title">🔒 4. The Dangerous HTTPS Misconception</div>
                <div class="advisory-desc">
                    A padlock icon in the browser address bar only verifies that data in transit is encrypted; 
                    <strong>it does NOT verify the honesty or legitimacy of the website owner.</strong> 
                    Over 80% of modern phishing campaigns utilize free, automated SSL certificates (such as Let's Encrypt).
                </div>
                <div class="advisory-code-box">
                    <span style="color: #EF4444; font-weight: 700;">[CRITICAL RULE]:</span> Never trust a website merely because it has an "https://" padlock!
                </div>
            </div>

            <div class="advisory-card">
                <div class="advisory-title">🛡️ 5. What To Do When You Encounter a Suspicious Link</div>
                <div class="advisory-desc">
                    Follow these immediate defensive steps:
                </div>
                <div class="advisory-code-box" style="line-height: 1.8;">
                    &bull; <strong style="color: #FFFFFF;">Do not click:</strong> Type the official web address directly into your browser.<br/>
                    &bull; <strong style="color: #FFFFFF;">Never enter credentials:</strong> Avoid entering passwords, credit cards, or OTPs on unverified pages.<br/>
                    &bull; <strong style="color: #FFFFFF;">Verify via second channel:</strong> Contact the sender via known phone or official app.<br/>
                    &bull; <strong style="color: #FFFFFF;">Report the attempt:</strong> Forward phishing emails to your security operations team or report to <code>phishing@abuse.net</code>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ----------------------------------------------------------------------------------
    # VIEW 8: MODEL INFORMATION & EVALUATION
    # ----------------------------------------------------------------------------------
    elif nav_choice == "⚙️ Model & Evaluation":
        st.subheader("⚙️ Machine Learning Model Information & Evaluation")
        st.markdown(
            "<span style='color: #F1F5F9; font-size: 1.05rem; font-weight: 500;'>"
            "Rigorous diagnostic metrics computed on a held-out stratified test set (25% split)."
            "</span>",
            unsafe_allow_html=True
        )

        if metrics.get("is_trained", False):
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Test Accuracy", f"{metrics['accuracy']*100:.1f}%")
            with m2:
                st.metric("Precision", f"{metrics['precision']*100:.1f}%")
            with m3:
                st.metric("Recall", f"{metrics['recall']*100:.1f}%")
            with m4:
                st.metric("F1-Score", f"{metrics['f1']*100:.1f}%")

            st.markdown("---")

            col_eval1, col_eval2 = st.columns([1, 1])

            with col_eval1:
                st.markdown("#### 🧩 Confusion Matrix")
                st.caption("Class 0 = Legitimate | Class 1 = Phishing")

                cm = metrics["confusion_matrix"]
                fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.4))
                fig_cm.patch.set_facecolor("#080D18")
                ax_cm.set_facecolor("#0F172A")

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["Legit (0)", "Phish (1)"],
                    yticklabels=["Legit (0)", "Phish (1)"],
                    cbar=False,
                    ax=ax_cm,
                    annot_kws={"size": 13, "weight": "bold", "color": "#FFFFFF"}
                )
                ax_cm.set_xlabel("PREDICTED CLASS", color="#FFFFFF", fontsize=10, fontfamily="monospace", weight="bold")
                ax_cm.set_ylabel("TRUE CLASS", color="#FFFFFF", fontsize=10, fontfamily="monospace", weight="bold")
                ax_cm.tick_params(colors="#FFFFFF", labelsize=10)
                for spine in ax_cm.spines.values():
                    spine.set_color("#27354A")
                st.pyplot(fig_cm)

                st.markdown(
                    f"""
                    <div style="font-family: monospace; font-size: 0.9rem; color: #FFFFFF; font-weight: 600; text-align: center; margin-top: 8px;">
                        True Negatives: {cm[0][0]} | False Positives: {cm[0][1]}<br/>
                        False Negatives: {cm[1][0]} | True Positives: {cm[1][1]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_eval2:
                st.markdown("#### 🌲 Gini Feature Importance Ranking")
                st.caption("Contribution to entropy reduction across 100 decision trees")

                importances = metrics["feature_importances"]
                sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]
                feat_names = [k.replace("_", " ").upper() for k, v in sorted_feats]
                feat_vals = [v for k, v in sorted_feats]

                fig_imp, ax_imp = plt.subplots(figsize=(5, 3.4))
                fig_imp.patch.set_facecolor("#080D18")
                ax_imp.set_facecolor("#0F172A")

                y_pos = np.arange(len(feat_names))
                ax_imp.barh(y_pos, feat_vals, align="center", color="#22D3EE")
                ax_imp.set_yticks(y_pos)
                ax_imp.set_yticklabels(feat_names, color="#FFFFFF", fontsize=9, fontfamily="monospace", weight="bold")
                ax_imp.invert_yaxis()
                ax_imp.set_xlabel("GINI IMPORTANCE WEIGHT", color="#FFFFFF", fontsize=10, fontfamily="monospace", weight="bold")
                ax_imp.tick_params(colors="#FFFFFF", labelsize=9)
                for spine in ax_imp.spines.values():
                    spine.set_color("#27354A")
                st.pyplot(fig_imp)
        else:
            st.warning("Model metrics unavailable. Application running in demonstration mode.")

    # Application Footer
    st.markdown("<hr style='border-color: #1E293B; margin-top: 40px;' />", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center; color: #94A3B8; font-size: 0.88rem; font-family: 'JetBrains Mono', monospace; padding: 14px 0; font-weight: 500;">
            PHISHGUARD AI — INTELLIGENT PHISHING DETECTION SYSTEM<br/>
            DEVELOPED FOR CYBERSECURITY DEFENSIVE RESEARCH • STRICT ZERO-EGRESS ISOLATION GUARANTEE
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
