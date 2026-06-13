import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Phishing Shield AI", page_icon="🛡️", layout="wide"
)

# Custom CSS for a modern, sleek look
st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        transform: translateY(-1px);
    }
    .metric-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------------
# 2. MOCK DATA GENERATION (Self-Contained Dataset)
# -------------------------------------------------------------------------
@st.cache_data
def load_mock_data():
    """Generates a balanced dataset of phishing and safe emails for training."""
    phishing_emails = [
        "URGENT: Your account has been suspended! Click here to verify your identity http://secure-bank-login-update.com",
        "Dear customer, we detected unusual activity on your Netflix account. Update your payment details immediately at http://netflix-verify-billing. net",
        "You have won a $1000 Amazon Gift Card! Claim your prize now by clicking this link http://free-rewards-center.com/win",
        "Official notice from PayPal: Your account is restricted. Please log in to http://paypal-security-checkup.org to resolve this.",
        "ATTENTION: Crypto wallet upgrade required. Failure to update via http://metamask-security-io.net will result in fund loss.",
        "Your package from FedEx could not be delivered. Address confirmation required at http://fedex-tracking-parcel.com",
        "Invoice payment overdue! Please review the attached document and click http://malicious-billing-portal.com to pay.",
        "Dear employee, your HR department requires you to review the new policy guidelines at http://internal-hr-portal-login.xyz",
    ] * 25  # Expand dataset

    safe_emails = [
        "Hi Team, just a reminder that our weekly sync meeting has been moved to Thursday at 10 AM. See you there!",
        "Your monthly electricity statement is now available online. Log into your standard portal to view your bill.",
        "Hey, are we still on for lunch tomorrow? Let me know if that Italian place down the street works for you.",
        "Thank you for your order! Your items have shipped and will arrive by Wednesday. Tracking number: 123456789.",
        "Hi John, I reviewed the project proposal and attached my notes. Let me know if you have any questions.",
        "Welcome to GitHub! Please confirm your email address to complete your account setup and start coding.",
        "Hi mom, just checking in to see how your doctor's appointment went today. Call me when you get a chance.",
        "Your weekly newsletter from Data Science Weekly is here. Top stories include ML advancements and Python tips.",
    ] * 25  # Expand dataset

    texts = phishing_emails + safe_emails
    # 1 for Phishing, 0 for Safe
    labels = [1] * len(phishing_emails) + [0] * len(safe_emails)

    return pd.DataFrame({"email_text": texts, "label": labels})


# -------------------------------------------------------------------------
# 3. FEATURE EXTRACTION & MODEL PIPELINE
# -------------------------------------------------------------------------
df = load_mock_data()

# Basic Heuristic Feature Engineering: Count URLs and Urgent Keywords
df["url_count"] = df["email_text"].apply(
    lambda x: x.count("http://") + x.count("https://") + x.count(".com")
)
df["urgent_count"] = df["email_text"].apply(
    lambda x: sum(
        1
        for w in ["urgent", "immediately", "suspended", "restricted", "verify"]
        if w in x.lower()
    )
)

# Text Vectorization using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
X_text = vectorizer.fit_transform(df["email_text"]).toarray()

# Combine TF-IDF text features with our engineered numeric features
X_numeric = df[["url_count", "urgent_count"]].values
X = np.hstack((X_text, X_numeric))
y = df["label"].values

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


# Helper function to extract features from user input dynamically
def extract_features_from_input(text):
    text_feat = vectorizer.transform([text]).toarray()
    urls = text.count("http://") + text.count("https://") + text.count(".com")
    urgents = sum(
        1
        for w in ["urgent", "immediately", "suspended", "restricted", "verify"]
        if w in text.lower()
    )
    num_feat = np.array([[urls, urgents]])
    return np.hstack((text_feat, num_feat))


# -------------------------------------------------------------------------
# 4. INTERACTIVE WEB INTERFACE (STREAMLIT UI)
# -------------------------------------------------------------------------

# Header Section
st.title("🛡️ Phishing Email Detection")
st.markdown(
    "Analyze incoming emails for malicious indicators using machine learning."
)
st.write("---")

# Layout: Split into Sidebar (Model Metrics) and Main Area (Live Tester)
col_metrics, col_tester = st.columns([1, 2], gap="large")

# --- LEFT COLUMN: MODEL PERFORMANCE ---
with col_metrics:
    st.subheader("📊 Model Diagnostics")

    # Display Accuracy Card
    st.markdown(
        f"""
        <div class="metric-box">
            <p style="color: #6B7280; margin-bottom: 0px; font-weight: 500;">MODEL ACCURACY</p>
            <h1 style="color: #4F46E5; margin-top: 0px;">{accuracy * 100:.1f}%</h1>
        </div>
    """,
        unsafe_allow_html=True,
)

    st.write("")

    # Visual Confusion Matrix using a stylized Pandas DataFrame
    st.markdown("**Confusion Matrix Matrix:**")
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Safe", "Actual Phishing"],
        columns=["Pred Safe", "Pred Phishing"],
    )
    st.dataframe(cm_df.style.background_gradient(cmap="Purples"), use_container_width=True)

    st.info(
        "💡 **How it works:** The model extracts TF-IDF text markers and counts suspicious indicators (like URLs and pressure words) to flag structural anomalies."
    )

# --- RIGHT COLUMN: LIVE TESTER ---
with col_tester:
    st.subheader("🔍 Live Email Inspector")
    st.write(
        "Paste the raw text of any suspicious email below to evaluate its risk score."
    )

    # Text Area Input
    user_input = st.text_area(
        "Email Content",
        height=200,
        placeholder="Paste email text here... (e.g., 'URGENT: Update your banking password immediately at http://fakebank.com')",
    )

    # Analyze Button
    if st.button("Analyze Email Risk"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some email content first!")
        else:
            # Process and predict
            features = extract_features_from_input(user_input)
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            phish_probability = probabilities[1] * 100

            # UI Feedback based on prediction
            st.write("---")
            if prediction == 1:
                st.error(f"🚨 **Result: PHISHING DETECTED**")
                st.progress(int(phish_probability))
                st.write(
                    f"Our system calculated a **{phish_probability:.1f}% risk factor**. This email contains language blueprints and URLs closely aligned with known fraud schemes."
                )
            else:
                st.success(f"✅ **Result: SAFE EMAIL**")
                st.progress(int(phish_probability))
                st.write(
                    f"Our system calculated a **{phish_probability:.1f}% risk factor**. This text lacks typical high-pressure keywords or deceptive hyperlinks."
                )

            # Detail Breakdown
            with st.expander("🔍 See Feature Breakdown"):
                urls_detected = user_input.count("http://") + user_input.count(
                    "https://"
                )
                st.write(f"- **Hyperlinks Detected:** {urls_detected}")
                st.write(
                    f"- **Urgency Keywords Flagged:** {sum(1 for w in ['urgent', 'immediately', 'suspended', 'restricted', 'verify'] if w in user_input.lower())}"
                )