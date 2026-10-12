"""Visual theme for the Crime Intelligence System (CSS + small HTML helpers).

Pairs with .streamlit/config.toml (dark base theme). Works without it, but looks best with it.
"""
import streamlit as st

ACCENT = "#4F8EF7"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1250px; }
#MainMenu, footer { visibility: hidden; }

/* page headings: accent bar */
.stApp h2 { border-left: 5px solid #4F8EF7; padding-left: 12px; font-weight: 700; letter-spacing: -0.3px; }
.stApp h3 { font-weight: 600; }

/* KPI cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(79,142,247,0.14), rgba(124,58,237,0.10));
    border: 1px solid rgba(79,142,247,0.30);
    border-radius: 16px; padding: 14px 18px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.18);
}
div[data-testid="stMetricValue"] { font-weight: 800; font-size: 1.9rem; }
div[data-testid="stMetricLabel"] p { opacity: 0.8; font-weight: 500; }

/* buttons */
.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
    border-radius: 12px; font-weight: 600; border: 1px solid rgba(79,142,247,0.5);
    transition: all .15s ease;
}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    border-color: #4F8EF7; box-shadow: 0 0 0 3px rgba(79,142,247,0.25); transform: translateY(-1px);
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1428 0%, #111c3a 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stRadio label { padding: 4px 0; }

/* alerts, tables, inputs */
div[data-testid="stAlert"] { border-radius: 14px; }
div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); }
div[data-baseweb="input"], div[data-baseweb="select"] > div { border-radius: 10px; }
div[data-testid="stForm"] { border-radius: 18px; border: 1px solid rgba(79,142,247,0.25); background: rgba(255,255,255,0.02); }
div[data-testid="stExpander"] { border-radius: 14px; }

/* hero banner */
.hero {
    background: radial-gradient(circle at 15% 20%, #3b82f6 0%, transparent 45%),
                linear-gradient(135deg, #1e1b4b 0%, #312e81 45%, #6d28d9 100%);
    border-radius: 22px; padding: 30px 32px; margin-bottom: 22px; color: #fff;
    box-shadow: 0 10px 40px rgba(79,70,229,0.35);
}
.hero h1 { margin: 0 0 6px 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px; color: #fff; border: none; padding: 0; }
.hero p  { margin: 0; font-size: 1.02rem; opacity: 0.88; color: #fff; }
.chip {
    display: inline-block; margin: 14px 8px 0 0; padding: 5px 13px; border-radius: 999px;
    background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.25);
    font-size: 0.8rem; font-weight: 600; color: #fff;
}

/* feature cards on login */
.fcard {
    border-radius: 16px; padding: 14px 16px; margin-bottom: 12px;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09);
}
.fcard b { font-size: 0.98rem; }
.fcard span { display: block; opacity: 0.72; font-size: 0.86rem; margin-top: 2px; }

/* sidebar user card */
.ucard {
    border-radius: 14px; padding: 12px 14px; margin-bottom: 10px;
    background: rgba(79,142,247,0.12); border: 1px solid rgba(79,142,247,0.30);
}
.ucard b { display: block; font-size: 1rem; }
.rolepill {
    display: inline-block; margin-top: 6px; padding: 2px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: .5px; text-transform: uppercase;
}
.role-police { background: #1d4ed8; color: #fff; }
.role-journalist { background: #b45309; color: #fff; }
.role-admin { background: #7c3aed; color: #fff; }

@media (max-width: 640px) {
    .hero { padding: 22px 20px; }
    .hero h1 { font-size: 1.55rem; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips=()):
    chip_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{chip_html}</div>', unsafe_allow_html=True)


def feature_card(title: str, text: str):
    st.markdown(f'<div class="fcard"><b>{title}</b><span>{text}</span></div>', unsafe_allow_html=True)


def user_card(name: str, role: str):
    st.markdown(f'<div class="ucard"><b>{name}</b><span class="rolepill role-{role}">{role}</span></div>',
                unsafe_allow_html=True)
