"""
Streamlit frontend for Flight Fare Prediction.

Provides a polished, professional UI to enter flight details and displays
the predicted fare by calling the FastAPI backend at http://localhost:8000.
"""

import os

import streamlit as st
import requests
from datetime import datetime, timedelta

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyFare BD – Flight Fare Predictor",
    page_icon="✈️",
    layout="centered",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ──────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero Banner ─────────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 40%, #2c5364 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.8rem;
    text-align: center;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 70%, rgba(255,255,255,0.04) 0%, transparent 60%);
}
.hero-banner h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.3rem;
    letter-spacing: -0.5px;
}
.hero-banner p {
    font-size: 0.95rem;
    opacity: 0.82;
    margin: 0;
    line-height: 1.5;
}
.hero-icon {
    font-size: 2.6rem;
    margin-bottom: 0.5rem;
    display: block;
}

/* ── Section Cards ───────────────────────────────────────────────────── */
.section-card {
    background: #ffffff;
    border: 1px solid #e8ecf1;
    border-radius: 14px;
    padding: 1.6rem 1.5rem 1.2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.section-card h3 {
    font-size: 0.92rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #2c5364;
    margin: 0 0 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #e8ecf1;
}

/* ── Result Card ─────────────────────────────────────────────────────── */
.result-card {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: white;
    margin: 1.4rem 0;
    box-shadow: 0 8px 32px rgba(15,32,39,0.25);
}
.result-card .fare-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    opacity: 0.7;
    margin-bottom: 0.3rem;
}
.result-card .fare-value {
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -1px;
    margin: 0.2rem 0 0;
}
.result-card .fare-currency {
    font-size: 1.1rem;
    font-weight: 400;
    opacity: 0.7;
    display: block;
    margin-top: 0.2rem;
}

/* ── Summary Pills ───────────────────────────────────────────────────── */
.summary-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
    justify-content: center;
    margin-top: 1rem;
}
.summary-pill {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 30px;
    padding: 0.45rem 1rem;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.9);
    backdrop-filter: blur(4px);
}
.summary-pill strong {
    color: white;
}

/* ── Detail Grid ─────────────────────────────────────────────────────── */
.detail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.8rem;
    margin-top: 1rem;
}
.detail-item {
    background: #f8fafc;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    border: 1px solid #eef1f5;
}
.detail-item .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #8899a6;
    margin-bottom: 0.2rem;
}
.detail-item .value {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1a2b3c;
}

/* ── Submit Button ───────────────────────────────────────────────────── */
div.stFormSubmitButton > button {
    background: linear-gradient(135deg, #2c5364 0%, #203a43 100%);
    color: white;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.5px;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 14px rgba(44,83,100,0.3);
}
div.stFormSubmitButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(44,83,100,0.4);
}
div.stFormSubmitButton > button:active {
    transform: translateY(0);
}

/* ── Footer ──────────────────────────────────────────────────────────── */
.app-footer {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    font-size: 0.78rem;
    color: #8899a6;
    border-top: 1px solid #eef1f5;
    margin-top: 2rem;
}
.app-footer .tech-badges {
    margin-top: 0.5rem;
}
.app-footer .badge {
    display: inline-block;
    background: #f0f4f8;
    color: #5a6f80;
    border-radius: 6px;
    padding: 0.2rem 0.55rem;
    font-size: 0.7rem;
    font-weight: 500;
    margin: 0.15rem 0.15rem;
}

/* ── Selectbox / Input refinements ───────────────────────────────────── */
div[data-baseweb="select"] {
    border-radius: 8px !important;
}
.stNumberInput input, .stDateInput input, .stTimeInput input {
    border-radius: 8px !important;
}

/* ── Hide default Streamlit branding ─────────────────────────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Fetch dropdown options from the API ─────────────────────────────────────
@st.cache_data(ttl=600)
def fetch_options():
    """Pull dropdown options from the /options endpoint."""
    try:
        resp = requests.get(f"{API_URL}/options", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {
            "airlines": [
                "Air Arabia", "Air Astra", "Air India", "AirAsia",
                "Biman Bangladesh Airlines", "British Airways", "Cathay Pacific",
                "Emirates", "Etihad Airways", "FlyDubai", "Gulf Air", "IndiGo",
                "Kuwait Airways", "Lufthansa", "Malaysian Airlines", "NovoAir",
                "Qatar Airways", "Saudia", "Singapore Airlines",
                "SriLankan Airlines", "Thai Airways", "Turkish Airlines",
                "US-Bangla Airlines", "Vistara",
            ],
            "sources": ["BZL", "CGP", "CXB", "DAC", "JSR", "RJH", "SPD", "ZYL"],
            "destinations": [
                "BKK", "BZL", "CCU", "CGP", "CXB", "DAC", "DEL", "DOH",
                "DXB", "IST", "JED", "JFK", "JSR", "KUL", "LHR", "RJH",
                "SIN", "SPD", "YYZ", "ZYL",
            ],
            "classes": ["Economy", "Business", "First Class"],
            "stopovers": ["Direct", "1 Stop", "2 Stops"],
            "aircraft_types": [
                "Airbus A320", "Airbus A350", "Boeing 737",
                "Boeing 777", "Boeing 787",
            ],
            "booking_sources": [
                "Direct Booking", "Online Website", "Travel Agency",
            ],
            "seasonalities": ["Regular", "Eid", "Hajj", "Winter Holidays"],
        }


options = fetch_options()

# Airport code → City name mapping for display
AIRPORT_NAMES = {
    "DAC": "Dhaka (DAC)", "CGP": "Chittagong (CGP)", "CXB": "Cox's Bazar (CXB)",
    "JSR": "Jessore (JSR)", "RJH": "Rajshahi (RJH)", "SPD": "Saidpur (SPD)",
    "ZYL": "Sylhet (ZYL)", "BZL": "Barisal (BZL)", "BKK": "Bangkok (BKK)",
    "CCU": "Kolkata (CCU)", "DEL": "Delhi (DEL)", "DOH": "Doha (DOH)",
    "DXB": "Dubai (DXB)", "IST": "Istanbul (IST)", "JED": "Jeddah (JED)",
    "JFK": "New York (JFK)", "KUL": "Kuala Lumpur (KUL)", "LHR": "London (LHR)",
    "SIN": "Singapore (SIN)", "YYZ": "Toronto (YYZ)",
}

def code_from_display(display_name: str) -> str:
    """Extract 3-letter IATA code from display name like 'Dhaka (DAC)'."""
    if "(" in display_name:
        return display_name.split("(")[-1].replace(")", "").strip()
    return display_name

# ── Hero Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <span class="hero-icon">✈️</span>
    <h1>SkyFare BD</h1>
    <p>Predict airfare for domestic &amp; international flights from Bangladesh.<br>
       Powered by machine learning on 57,000+ real flight records.</p>
</div>
""", unsafe_allow_html=True)

# ── Input form ──────────────────────────────────────────────────────────────
with st.form("flight_form"):

    # ── Section 1 : Route & Airline ─────────────────────────────────────
    st.markdown('<div class="section-card"><h3>🛫  Route & Airline</h3>',
                unsafe_allow_html=True)
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        source_display = [AIRPORT_NAMES.get(s, s) for s in options["sources"]]
        source_sel = st.selectbox("From", source_display,
                                  index=source_display.index(AIRPORT_NAMES.get("DAC", "DAC")))
    with r1c2:
        dest_display = [AIRPORT_NAMES.get(d, d) for d in options["destinations"]]
        dest_sel = st.selectbox("To", dest_display,
                                index=dest_display.index(AIRPORT_NAMES.get("CXB", "CXB")))
    with r1c3:
        airline = st.selectbox("Airline", options["airlines"], index=4)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 2 : Schedule ────────────────────────────────────────────
    st.markdown('<div class="section-card"><h3>📅  Schedule</h3>',
                unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        departure_date = st.date_input(
            "Departure Date",
            value=datetime.now().date() + timedelta(days=15),
        )
    with s2:
        departure_time = st.time_input(
            "Departure Time",
            value=datetime.strptime("08:30", "%H:%M").time(),
        )
    with s3:
        days_before = st.number_input(
            "Days Before Departure", min_value=0, max_value=365,
            value=15, step=1,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 3 : Flight Details ──────────────────────────────────────
    st.markdown('<div class="section-card"><h3>🛩️  Flight Details</h3>',
                unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        travel_class = st.selectbox("Travel Class", options["classes"])
    with d2:
        stopovers = st.selectbox("Stopovers", options["stopovers"])
    with d3:
        aircraft_type = st.selectbox("Aircraft Type", options["aircraft_types"])
    with d4:
        duration_hrs = st.number_input(
            "Duration (hrs)", min_value=0.1, max_value=30.0,
            value=1.5, step=0.5,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 4 : Booking Info ────────────────────────────────────────
    st.markdown('<div class="section-card"><h3>🎫  Booking Info</h3>',
                unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        booking_source = st.selectbox("Booking Source", options["booking_sources"], index=1)
    with b2:
        seasonality = st.selectbox("Seasonality / Event", options["seasonalities"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Submit ──────────────────────────────────────────────────────────
    submitted = st.form_submit_button("✈️  Predict Fare", use_container_width=True)

# ── Prediction ──────────────────────────────────────────────────────────────
if submitted:
    source = code_from_display(source_sel)
    destination = code_from_display(dest_sel)

    departure_datetime = (
        f"{departure_date.strftime('%Y-%m-%d')} "
        f"{departure_time.strftime('%H:%M:%S')}"
    )

    payload = {
        "airline": airline,
        "source": source,
        "destination": destination,
        "departure_datetime": departure_datetime,
        "duration_hrs": duration_hrs,
        "stopovers": stopovers,
        "aircraft_type": aircraft_type,
        "travel_class": travel_class,
        "booking_source": booking_source,
        "seasonality": seasonality,
        "days_before_departure": days_before,
    }

    with st.spinner("Calculating your fare..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            fare = result["predicted_fare_bdt"]
            summary = result["input_summary"]

            # ── Result Card ─────────────────────────────────────────
            st.markdown(f"""
            <div class="result-card">
                <div class="fare-label">Estimated Total Fare</div>
                <div class="fare-value">৳ {fare:,.0f}</div>
                <span class="fare-currency">Bangladeshi Taka (BDT)</span>
                <div class="summary-row">
                    <span class="summary-pill">🛫 <strong>{summary['route']}</strong></span>
                    <span class="summary-pill">🏢 <strong>{summary['airline']}</strong></span>
                    <span class="summary-pill">💺 <strong>{summary['class']}</strong></span>
                    <span class="summary-pill">⏱️ <strong>{duration_hrs}h</strong></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Flight Details Grid ─────────────────────────────────
            dep_raw = summary["departure"].replace(" ", "T")
            dep_dt = datetime.strptime(dep_raw, "%Y-%m-%dT%H:%M:%S")
            dep_formatted = dep_dt.strftime("%b %d, %Y  •  %I:%M %p")

            st.markdown(f"""
            <div class="section-card" style="margin-top: 0.5rem;">
                <h3>📋  Booking Details</h3>
                <div class="detail-grid">
                    <div class="detail-item">
                        <div class="label">Departure</div>
                        <div class="value">{dep_formatted}</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">Booked</div>
                        <div class="value">{summary['days_before']} days in advance</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">Stopovers</div>
                        <div class="value">{stopovers}</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">Aircraft</div>
                        <div class="value">{aircraft_type}</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">Booking Source</div>
                        <div class="value">{booking_source}</div>
                    </div>
                    <div class="detail-item">
                        <div class="label">Season</div>
                        <div class="value">{seasonality}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except requests.exceptions.ConnectionError:
            st.error(
                "🚫 Cannot reach the API server. Please make sure the FastAPI "
                f"backend is running at **{API_URL}**."
            )
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    SkyFare BD — Flight Fare Prediction System<br>
    <div class="tech-badges">
        <span class="badge">CatBoost</span>
        <span class="badge">R² ≈ 0.69</span>
        <span class="badge">MAE ≈ 30K BDT</span>
        <span class="badge">FastAPI</span>
        <span class="badge">Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)
