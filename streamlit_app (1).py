"""
=============================================================================
 BULKOWSKI + DONNELLY — STREAMLIT MOBILE APP
 -----------------------------------------------------------------------------
 Mobile-first UI for the pattern detection engine. Deploys to Streamlit Cloud.

 Features:
   - CSV upload OR Angel One live data fetch
   - All 5 engines: Bulkowski / Brooks PA / Street Smarts / Donnelly / Quant
   - Plotly interactive chart (zoom/pan on mobile)
   - Backtesting (with progress + caching)
   - Pattern forecasting
   - Mobile-optimized layout: single column, large tap targets, collapsible sections

 Setup:
   pip install streamlit pandas numpy scipy plotly smartapi-python pyotp websocket-client
   streamlit run streamlit_app.py

 Deploy to Streamlit Cloud:
   1. Push streamlit_app.py + engine.py + requirements.txt to a public GitHub repo
   2. Go to share.streamlit.io and link your repo
   3. (Optional) Add Angel One credentials in Streamlit Secrets for auto-login
=============================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime as _dt
from io import BytesIO

# Force matplotlib Agg backend BEFORE importing engine (no display on cloud)
import os
os.environ.setdefault('MPLBACKEND', 'Agg')

# Import the engine module (Bulkowski + Donnelly + all detectors)
import engine

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — mobile-first
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bulkowski + Donnelly Pattern Analyzer",
    page_icon="📊",
    layout="centered",          # 'centered' = mobile-friendly narrow column
    initial_sidebar_state="collapsed",
)

# Custom CSS for mobile compatibility
st.markdown("""
<style>
    /* Reduce padding on small screens */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    /* Bigger tap targets for buttons */
    .stButton > button {
        width: 100%;
        padding: 0.6rem 1rem;
        font-size: 1rem;
        border-radius: 8px;
    }
    /* Compact metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.1rem;
    }
    /* Make text more readable on small screens */
    .signal-card {
        background: #f0f4f8;
        border-left: 4px solid #1e6fa8;
        padding: 10px 12px;
        margin: 6px 0;
        border-radius: 6px;
    }
    .bullish { border-left-color: #2ecc71 !important; background: #e8f8ee; }
    .bearish { border-left-color: #e74c3c !important; background: #fdebea; }
    .neutral { border-left-color: #f39c12 !important; background: #fef5e6; }
    /* Tabs more touchable */
    button[role="tab"] {
        font-size: 0.95rem !important;
        padding: 0.5rem 0.8rem !important;
    }
    /* Hide Streamlit decorations on mobile */
    @media (max-width: 640px) {
        header, footer { display: none !important; }
        #MainMenu { display: none !important; }
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if 'df' not in st.session_state:
    st.session_state.df = None
if 'detected' not in st.session_state:
    st.session_state.detected = []
if 'brooks' not in st.session_state:
    st.session_state.brooks = []
if 'streets' not in st.session_state:
    st.session_state.streets = []
if 'donnelly' not in st.session_state:
    st.session_state.donnelly = []
if 'quant' not in st.session_state:
    st.session_state.quant = []
if 'angel_obj' not in st.session_state:
    st.session_state.angel_obj = None
if 'angel_jwt' not in st.session_state:
    st.session_state.angel_jwt = None
if 'data_label' not in st.session_state:
    st.session_state.data_label = ""


# ─────────────────────────────────────────────────────────────────────────────
# ANGEL ONE — standalone helpers (extracted from engine class methods)
# ─────────────────────────────────────────────────────────────────────────────
ANGEL_TOKENS = {
    'NIFTY':       ('NSE', '99926000'),
    'NIFTY 50':    ('NSE', '99926000'),
    'BANKNIFTY':   ('NSE', '99926009'),
    'BANK NIFTY':  ('NSE', '99926009'),
    'FINNIFTY':    ('NSE', '99926037'),
    'MIDCPNIFTY':  ('NSE', '99926012'),
    'SENSEX':      ('BSE', '99919000'),
    'INDIA VIX':   ('NSE', '99926017'),
    'RELIANCE':    ('NSE', '2885'),
    'TCS':         ('NSE', '11536'),
    'HDFCBANK':    ('NSE', '1333'),
    'INFY':        ('NSE', '1594'),
    'ICICIBANK':   ('NSE', '4963'),
    'SBIN':        ('NSE', '3045'),
    'BHARTIARTL':  ('NSE', '10604'),
    'KOTAKBANK':   ('NSE', '1922'),
    'AXISBANK':    ('NSE', '5900'),
    'BAJFINANCE':  ('NSE', '317'),
    'TATAMOTORS':  ('NSE', '3456'),
    'TATASTEEL':   ('NSE', '3499'),
    'WIPRO':       ('NSE', '3787'),
    'HCLTECH':     ('NSE', '7229'),
    'MARUTI':      ('NSE', '10999'),
    'ITC':         ('NSE', '1660'),
}

ANGEL_INTERVALS = {
    '1 Min':   'ONE_MINUTE',
    '3 Min':   'THREE_MINUTE',
    '5 Min':   'FIVE_MINUTE',
    '10 Min':  'TEN_MINUTE',
    '15 Min':  'FIFTEEN_MINUTE',
    '30 Min':  'THIRTY_MINUTE',
    '1 Hour':  'ONE_HOUR',
    '1 Day':   'ONE_DAY',
    '1 Week':  'ONE_WEEK',
}


def angel_connect(api_key, client_id, password, totp_key):
    """Connect to Angel One. Returns (smartapi_obj, jwt, message)."""
    try:
        from SmartApi import SmartConnect
        import pyotp
    except ImportError:
        return None, None, "❌ smartapi-python not installed. Add to requirements.txt: smartapi-python pyotp"

    try:
        obj = SmartConnect(api_key=api_key)
        if not totp_key or not totp_key.strip():
            return None, None, "❌ TOTP secret key required"
        totp_code = pyotp.TOTP(totp_key.strip()).now()
        data = obj.generateSession(client_id, password, totp_code)
        if not data or not data.get('status'):
            msg = data.get('message', 'Unknown error') if data else 'No response'
            return None, None, f"❌ Login failed: {msg}"

        sd = data.get('data', {})
        jwt = sd.get('jwtToken', '')
        if jwt.startswith('Bearer '):
            jwt = jwt[7:]
        feed = sd.get('feedToken', '')
        obj.setAccessToken(jwt)
        try: obj.setFeedToken(feed)
        except Exception: pass
        try: obj.setUserId(client_id)
        except Exception: pass
        return obj, jwt, f"✅ Connected as {client_id}"
    except Exception as e:
        return None, None, f"❌ Connection error: {e}"


def angel_fetch_ohlc(jwt, api_key, symbol, interval_label, from_date, to_date):
    """
    Fetch OHLC via direct HTTP. Returns DataFrame with columns
    Date, Open, High, Low, Close, Volume (or None on failure).
    """
    import urllib.request, json as _json, ssl

    symbol_upper = symbol.strip().upper()
    interval     = ANGEL_INTERVALS.get(interval_label, 'ONE_DAY')
    is_intraday  = interval not in ('ONE_DAY', 'ONE_WEEK', 'ONE_MONTH')

    # Normalize dates to strict YYYY-MM-DD
    from_date = _dt.datetime.strptime(str(from_date).strip(), '%Y-%m-%d').strftime('%Y-%m-%d')
    to_date   = _dt.datetime.strptime(str(to_date).strip(),   '%Y-%m-%d').strftime('%Y-%m-%d')
    from_str  = f"{from_date} 09:15" if is_intraday else f"{from_date} 09:00"
    to_str    = f"{to_date} 15:30"   if is_intraday else f"{to_date} 15:30"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE

    headers = {
        'Content-Type':    'application/json',
        'Accept':          'application/json',
        'X-UserType':      'USER',
        'X-SourceID':      'WEB',
        'X-ClientLocalIP': '127.0.0.1',
        'X-ClientPublicIP':'127.0.0.1',
        'X-MACAddress':    '00:00:00:00:00:00',
        'X-PrivateKey':    api_key,
        'Authorization':   f'Bearer {jwt}',
    }

    # Use preset token map first
    preset = ANGEL_TOKENS.get(symbol_upper)
    if preset:
        exchange, token = preset
    else:
        # Search via API
        try:
            search_url  = ('https://apiconnect.angelone.in/rest/secure/'
                           'angelbroking/order/v1/searchScrip')
            search_body = _json.dumps({'exchange':'NSE', 'searchscrip':symbol_upper}).encode()
            req  = urllib.request.Request(search_url, data=search_body, headers=headers, method='POST')
            resp = urllib.request.urlopen(req, timeout=10, context=ctx)
            d = _json.loads(resp.read())
            if d.get('status') and d.get('data'):
                first = d['data'][0]
                exchange = first.get('exchange', 'NSE')
                token    = first.get('symboltoken', '')
            else:
                return None
        except Exception:
            return None

    if not token:
        return None

    # Fetch candles
    url = ('https://apiconnect.angelone.in/rest/secure/angelbroking/'
           'historical/v1/getCandleData')
    body = _json.dumps({
        'exchange':    exchange,
        'symboltoken': str(token),
        'interval':    interval,
        'fromdate':    from_str,
        'todate':      to_str,
    }).encode()
    try:
        req  = urllib.request.Request(url, data=body, headers=headers, method='POST')
        resp = urllib.request.urlopen(req, timeout=20, context=ctx)
        d    = _json.loads(resp.read())
        if not d.get('status') or not d.get('data'):
            return None
        rows = d['data']
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=['Date','Open','High','Low','Close','Volume'])
        df = df.astype({'Open':float,'High':float,'Low':float,
                        'Close':float,'Volume':float})
        df['Date'] = pd.to_datetime(df['Date'])
        return df.reset_index(drop=True)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CHART RENDERING — Plotly mobile-friendly
# ─────────────────────────────────────────────────────────────────────────────
def render_plotly_chart(df, detected, brooks, streets, donnelly, quant, height=520):
    """Build a mobile-friendly interactive Plotly candlestick chart."""
    if df is None or len(df) == 0:
        return None

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
        subplot_titles=("", "Volume"),
    )

    # ── Candlesticks ──────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
        name='Price',
        showlegend=False,
    ), row=1, col=1)

    # ── Volume bars ───────────────────────────────────────────────────────
    if 'Volume' in df.columns:
        colors = ['#26a69a' if c >= o else '#ef5350'
                  for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(
            x=df['Date'], y=df['Volume'],
            marker_color=colors, name='Volume',
            showlegend=False,
        ), row=2, col=1)

    # ── Pattern overlays ──────────────────────────────────────────────────
    import re as _re

    def _extract_price(s):
        nums = _re.findall(r'\d+\.?\d*', str(s))
        return float(nums[0]) if nums else None

    # Bulkowski patterns — gold lines
    for p in (detected or [])[:5]:
        if 'neckline' in p:
            nl = p['neckline']
            fig.add_hline(y=nl, line=dict(color='#f1c40f', width=2),
                          row=1, col=1,
                          annotation_text=f"BK: {p.get('name','')[:18]} ({nl:.0f})",
                          annotation_position='right',
                          annotation_font=dict(color='#b8860b', size=9))

    # Brooks PA — teal dashed
    for s in (brooks or [])[:3]:
        ev = _extract_price(s.get('entry', ''))
        if ev:
            fig.add_hline(y=ev, line=dict(color='#1abc9c', width=1.5, dash='dash'),
                          row=1, col=1,
                          annotation_text=f"PA: {s['name'][8:][:14]} ({ev:.0f})",
                          annotation_position='right',
                          annotation_font=dict(color='#16a085', size=8))

    # Street Smarts — sienna dotted
    for s in (streets or [])[:3]:
        ev = _extract_price(s.get('entry', ''))
        if ev:
            fig.add_hline(y=ev, line=dict(color='#cd853f', width=1.5, dash='dot'),
                          row=1, col=1,
                          annotation_text=f"SS: {s['name'][15:][:14]} ({ev:.0f})",
                          annotation_position='right',
                          annotation_font=dict(color='#8b4513', size=8))

    # Donnelly — royal blue dot-dash
    for s in (donnelly or [])[:3]:
        ev = _extract_price(s.get('entry', ''))
        if ev:
            nm = s['name'].replace('Donnelly: ', '')[:14]
            fig.add_hline(y=ev, line=dict(color='#1e6fa8', width=1.5, dash='dashdot'),
                          row=1, col=1,
                          annotation_text=f"DN: {nm} ({ev:.0f})",
                          annotation_position='right',
                          annotation_font=dict(color='#1e6fa8', size=8))

    # ── Layout — mobile-optimized ─────────────────────────────────────────
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        dragmode='pan',
        font=dict(size=10),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
    )
    fig.update_xaxes(
        showgrid=True, gridcolor='#e8e8e8',
        rangeslider_visible=False,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor='#e8e8e8',
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL CARD RENDERERS
# ─────────────────────────────────────────────────────────────────────────────
def render_signal_card(s, kind='generic'):
    """Render a single detected signal as an expandable card."""
    icon = s.get('icon', '◆')
    name = s.get('name', 'Signal')
    conf = s.get('confidence', 0)
    sig_text = s.get('signal', '')

    direction = 'neutral'
    if '▲' in sig_text or 'BULL' in sig_text.upper() or 'BUY' in sig_text.upper() or 'LONG' in sig_text.upper():
        direction = 'bullish'
    elif '▼' in sig_text or 'BEAR' in sig_text.upper() or 'SELL' in sig_text.upper() or 'SHORT' in sig_text.upper():
        direction = 'bearish'

    direction_emoji = {'bullish': '🟢', 'bearish': '🔴', 'neutral': '🟡'}[direction]

    with st.expander(f"{direction_emoji} {icon} **{name}** — {conf:.0f}%", expanded=False):
        if s.get('source'):
            st.caption(f"📖 {s['source']}")

        st.markdown(f"**Signal:** {sig_text}")

        # Confidence bar
        st.progress(min(1.0, conf / 100), text=f"Confidence: {conf:.0f}%")

        if s.get('value'):
            st.markdown(f"**Reading:** {s['value']}")

        # Trading plan
        st.markdown("---")
        st.markdown("**📋 TRADING PLAN**")
        if s.get('entry'):
            st.success(f"▸ **ENTRY:** {s['entry']}")
        if s.get('stop'):
            st.error(f"▸ **STOP:** {s['stop']}")
        if s.get('target'):
            st.warning(f"▸ **TARGET:** {s['target']}")

        # R:R if numbers can be parsed
        try:
            import re as _re
            def ep(x):
                nums = _re.findall(r'\d+\.?\d*', str(x))
                return float(nums[0]) if nums else None
            ev = ep(s.get('entry'))
            sv = ep(s.get('stop'))
            tv = ep(s.get('target'))
            if ev and sv and tv:
                rr = engine.calc_rr(ev, sv, tv)
                if rr:
                    grade_color = '🟢' if rr['rr'] >= 2 else ('🟡' if rr['rr'] >= 1 else '🔴')
                    st.markdown(f"{grade_color} **R:R:** 1:{rr['rr']:.1f} | "
                                f"Risk: {rr['risk']:.2f} | Reward: {rr['reward']:.2f} | "
                                f"Grade: **{rr['grade']}**")
                    st.caption(rr['advice'])
        except Exception:
            pass

        # Description / explanation
        if s.get('desc'):
            st.markdown("---")
            st.markdown("**💡 EXPLANATION**")
            st.markdown(s['desc'])


def render_bulkowski_card(p):
    """Bulkowski patterns have a different structure — render distinctly."""
    name = p.get('name', 'Pattern')
    conf = p.get('confidence', 0)
    direction = p.get('direction', 'NEUTRAL').upper()

    icon = '▲' if 'BULL' in direction else ('▼' if 'BEAR' in direction else '◆')
    emoji = {'BULL': '🟢', 'BEAR': '🔴'}.get(direction.split()[0] if direction else '', '🟡')

    with st.expander(f"{emoji} {icon} **{name}** — {conf:.0f}%", expanded=False):
        # Try to look up in PATTERNS_DB for full info
        db = engine.PATTERNS_DB.get(name, {})
        if db.get('description'):
            st.caption(db['description'])

        cols = st.columns(2)
        cols[0].metric("Direction", direction)
        cols[1].metric("Confidence", f"{conf:.0f}%")

        # Completion status
        comp = p.get('completion', {})
        if comp.get('status'):
            st.info(f"📍 {comp['status']}: {comp.get('description', '')}")

        # Bulkowski stats (from DB)
        if db.get('stats'):
            st.markdown("**📊 BULKOWSKI STATISTICS**")
            for market_kind, stats in db['stats'].items():
                with st.container():
                    st.markdown(f"*{market_kind.replace('_',' ').title()}:*")
                    sc1, sc2 = st.columns(2)
                    if 'avg_rise' in stats: sc1.metric("Avg Rise", stats['avg_rise'])
                    if 'avg_decline' in stats: sc1.metric("Avg Decline", stats['avg_decline'])
                    if 'breakeven_failure_rate' in stats: sc2.metric("Breakeven Fail", stats['breakeven_failure_rate'])
                    if 'samples' in stats: sc2.metric("Samples", str(stats['samples']))

        # Trading plan
        tp = db.get('trading_plan', {})
        if tp:
            st.markdown("---")
            st.markdown("**📋 TRADING PLAN**")
            if tp.get('entry'):  st.success(f"▸ **ENTRY:** {tp['entry']}")
            if tp.get('stop'):   st.error(f"▸ **STOP:** {tp['stop']}")
            if tp.get('target'): st.warning(f"▸ **TARGET:** {tp['target']}")

        # Identification rules
        if db.get('identification'):
            st.markdown("---")
            st.markdown("**🔍 HOW TO IDENTIFY**")
            for rule in db['identification']:
                st.markdown(f"- {rule}")

        # Measure rule
        if db.get('measure_rule'):
            st.markdown("---")
            st.info(f"📏 **MEASURE RULE:** {db['measure_rule']}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 Pattern Analyzer")
st.caption("Bulkowski · Brooks · Street Smarts · Donnelly · Quant — Mobile Edition")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING TABS
# ─────────────────────────────────────────────────────────────────────────────
data_tab1, data_tab2 = st.tabs(["📁 Upload CSV", "🔌 Angel One Live"])

with data_tab1:
    st.markdown("Upload a CSV with columns: **Date, Open, High, Low, Close, Volume**")
    uploaded = st.file_uploader("Choose CSV file", type=['csv'], key='csv_upload',
                                label_visibility='collapsed')
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            # Normalize columns case-insensitively
            cols_map = {c.lower(): c for c in df.columns}
            for need in ['date','open','high','low','close']:
                if need not in cols_map:
                    st.error(f"❌ Missing column: {need}")
                    df = None
                    break
            if df is not None:
                rename = {cols_map[k.lower()]: k.title() for k in ['date','open','high','low','close']
                          if k.lower() in cols_map}
                if 'volume' in cols_map:
                    rename[cols_map['volume']] = 'Volume'
                df = df.rename(columns=rename)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date']).reset_index(drop=True)
                if len(df) < 10:
                    st.error("❌ Need at least 10 rows of data")
                else:
                    st.session_state.df = df
                    st.session_state.data_label = uploaded.name
                    st.session_state.detected = []
                    st.session_state.brooks = []
                    st.session_state.streets = []
                    st.session_state.donnelly = []
                    st.session_state.quant = []
                    st.success(f"✅ Loaded {len(df)} bars: {df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()}")
        except Exception as e:
            st.error(f"❌ Could not parse CSV: {e}")

with data_tab2:
    st.markdown("**Angel One SmartAPI — Live Data**")

    # ── Local-storage credential save/load ────────────────────────────────
    # Credentials live in YOUR browser's localStorage, encrypted with a
    # passphrase you choose. They never leave your device.
    try:
        from streamlit_local_storage import LocalStorage
        ls = LocalStorage()
        _LS_OK = True
    except ImportError:
        ls = None
        _LS_OK = False

    def _xor_encrypt(plaintext: str, key: str) -> str:
        """Simple XOR + base64. Lightweight obfuscation, not crypto-grade."""
        import base64
        if not plaintext or not key:
            return ''
        kb = key.encode('utf-8')
        pb = plaintext.encode('utf-8')
        x = bytes(b ^ kb[i % len(kb)] for i, b in enumerate(pb))
        return base64.b64encode(x).decode('ascii')

    def _xor_decrypt(ciphertext: str, key: str) -> str:
        import base64
        if not ciphertext or not key:
            return ''
        try:
            x = base64.b64decode(ciphertext.encode('ascii'))
            kb = key.encode('utf-8')
            return bytes(b ^ kb[i % len(kb)] for i, b in enumerate(x)).decode('utf-8')
        except Exception:
            return ''

    LS_KEY_CREDS    = 'angel_creds_v1'
    LS_KEY_HASMARK  = 'angel_creds_marker'   # tells us if creds exist

    # Try to read defaults from Streamlit Secrets if available
    secrets_available = False
    try:
        if hasattr(st, 'secrets') and 'angel_one' in st.secrets:
            secrets_available = True
    except Exception:
        pass

    # ── Saved-credentials unlock UI ────────────────────────────────────────
    saved_blob = None
    has_saved = False
    if _LS_OK:
        try:
            saved_blob = ls.getItem(LS_KEY_CREDS)
            has_saved = bool(ls.getItem(LS_KEY_HASMARK))
        except Exception:
            saved_blob = None
            has_saved = False

    if has_saved:
        with st.expander("🔓 Saved credentials found — unlock", expanded=True):
            col_u1, col_u2 = st.columns([3, 1])
            with col_u1:
                unlock_pass = st.text_input(
                    "Passphrase",
                    type='password',
                    key='unlock_pass',
                    label_visibility='collapsed',
                    placeholder="Enter passphrase to unlock")
            with col_u2:
                unlock_clicked = st.button("Unlock", use_container_width=True, key='unlock_btn')
            col_x1, col_x2 = st.columns(2)
            with col_x1:
                if st.button("🗑️ Clear saved", use_container_width=True, key='clear_creds'):
                    if _LS_OK:
                        try:
                            ls.deleteItem(LS_KEY_CREDS)
                            ls.deleteItem(LS_KEY_HASMARK)
                        except Exception:
                            pass
                    st.success("Cleared. Reload the page.")
                    st.stop()

            if unlock_clicked and unlock_pass and saved_blob:
                import json as _json
                decrypted = _xor_decrypt(saved_blob, unlock_pass)
                try:
                    creds = _json.loads(decrypted) if decrypted else {}
                    if creds.get('client_id'):
                        st.session_state['preload_creds'] = creds
                        st.success("✅ Credentials loaded — login form pre-filled below")
                    else:
                        st.error("❌ Wrong passphrase")
                except Exception:
                    st.error("❌ Wrong passphrase")

    # Decide what to pre-fill the form with: secrets > preloaded > empty
    preload = st.session_state.get('preload_creds', {})
    def _default(field):
        if preload.get(field):
            return preload[field]
        if secrets_available:
            return st.secrets.get('angel_one', {}).get(field, '')
        return ''

    # ── Login form ─────────────────────────────────────────────────────────
    with st.form("angel_login_form", clear_on_submit=False):
        api_key = st.text_input("API Key",
            value=_default('api_key'),
            type='password')
        client_id = st.text_input("Client ID",
            value=_default('client_id'))
        password = st.text_input("Password",
            value=_default('password'),
            type='password')
        totp_key = st.text_input("TOTP Secret Key",
            value=_default('totp_key'),
            type='password',
            help="The secret key from your TOTP app — not the 6-digit code")

        # Save-credentials section inside the form
        save_creds = st.checkbox(
            "💾 Save credentials in this browser (encrypted)",
            value=False,
            disabled=not _LS_OK,
            help=("Saves to your browser's local storage, encrypted with a "
                  "passphrase you choose. Never leaves your device.")
                  if _LS_OK else
                 "Requires: pip install streamlit-local-storage")
        save_pass = ''
        if save_creds and _LS_OK:
            save_pass = st.text_input(
                "Choose a passphrase to encrypt with",
                type='password',
                help="You'll enter this on next visit to unlock saved credentials")

        submit_login = st.form_submit_button("🔌 Connect", use_container_width=True)

    if submit_login:
        with st.spinner("Connecting..."):
            obj, jwt, msg = angel_connect(api_key, client_id, password, totp_key)
            if obj:
                st.session_state.angel_obj = obj
                st.session_state.angel_jwt = jwt
                st.session_state.angel_api_key = api_key
                st.success(msg)

                # Save credentials if requested
                if save_creds and _LS_OK and save_pass:
                    import json as _json
                    creds_blob = _json.dumps({
                        'api_key': api_key,
                        'client_id': client_id,
                        'password': password,
                        'totp_key': totp_key,
                    })
                    try:
                        ls.setItem(LS_KEY_CREDS, _xor_encrypt(creds_blob, save_pass))
                        ls.setItem(LS_KEY_HASMARK, '1')
                        st.info("🔒 Credentials saved in this browser. "
                                "Use the same passphrase to unlock on next visit.")
                    except Exception as e:
                        st.warning(f"Could not save: {e}")
                elif save_creds and not _LS_OK:
                    st.warning("⚠️ Save requested but `streamlit-local-storage` not installed. "
                               "Add it to requirements.txt and redeploy.")
                elif save_creds and not save_pass:
                    st.warning("⚠️ Enter a passphrase to save credentials")
            else:
                st.error(msg)

    if st.session_state.angel_obj:
        st.markdown("---")
        st.markdown("**📥 Fetch Data**")

        col1, col2 = st.columns(2)
        with col1:
            symbol = st.selectbox("Symbol",
                options=list(ANGEL_TOKENS.keys()),
                index=list(ANGEL_TOKENS.keys()).index('NIFTY'))
        with col2:
            interval_label = st.selectbox("Interval",
                options=list(ANGEL_INTERVALS.keys()),
                index=list(ANGEL_INTERVALS.keys()).index('1 Day'))

        col3, col4 = st.columns(2)
        with col3:
            from_date = st.date_input("From",
                value=_dt.date.today() - _dt.timedelta(days=180))
        with col4:
            to_date = st.date_input("To", value=_dt.date.today())

        if st.button("📥 Fetch OHLC", use_container_width=True, type='primary'):
            with st.spinner(f"Fetching {symbol} {interval_label}..."):
                df = angel_fetch_ohlc(
                    st.session_state.angel_jwt,
                    st.session_state.angel_api_key,
                    symbol, interval_label,
                    str(from_date), str(to_date))
            if df is not None and len(df) > 0:
                st.session_state.df = df
                st.session_state.data_label = f"{symbol} {interval_label}"
                st.session_state.detected = []
                st.session_state.brooks = []
                st.session_state.streets = []
                st.session_state.donnelly = []
                st.session_state.quant = []
                st.success(f"✅ Fetched {len(df)} bars")
            else:
                st.error("❌ No data returned. Check date range / symbol / token validity.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS — only show when data is loaded
# ─────────────────────────────────────────────────────────────────────────────
df = st.session_state.df
if df is None:
    st.info("⬆️ Upload a CSV or fetch data via Angel One to begin analysis")
    st.stop()

st.markdown("---")
st.markdown(f"### 📈 {st.session_state.data_label}")

# Quick stats
sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Bars", f"{len(df)}")
sc2.metric("Last Close", f"{df['Close'].iloc[-1]:,.2f}")
prev_close = df['Close'].iloc[-2] if len(df) >= 2 else df['Close'].iloc[-1]
chg = df['Close'].iloc[-1] - prev_close
chg_pct = (chg / prev_close * 100) if prev_close else 0
sc3.metric("Change", f"{chg:+.2f}", f"{chg_pct:+.2f}%")
sc4.metric("Range", f"{df['Low'].min():.0f}–{df['High'].max():.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# DETECT BUTTON
# ─────────────────────────────────────────────────────────────────────────────
if st.button("🔍 RUN DETECTION", use_container_width=True, type='primary'):
    progress = st.progress(0, text="Running detectors...")
    try:
        progress.progress(10, text="Bulkowski patterns...")
        detected = engine.detect_patterns(df) if len(df) >= 40 else []
        for p in detected:
            if 'completion' not in p:
                p['completion'] = engine.get_completion_status(p, df)

        progress.progress(35, text="Brooks Price Action...")
        brooks = engine.detect_brooks_signals(df) if len(df) >= 10 else []

        progress.progress(55, text="Street Smarts...")
        streets = engine.detect_street_smarts(df) if len(df) >= 22 else []

        progress.progress(75, text="Donnelly Setups...")
        donnelly = engine.detect_donnelly(df) if len(df) >= 22 else []

        progress.progress(90, text="Quant Signals...")
        quant = engine.compute_quant_signals(df) if len(df) >= 20 else []

        progress.progress(100, text="Done!")

        st.session_state.detected = detected
        st.session_state.brooks = brooks
        st.session_state.streets = streets
        st.session_state.donnelly = donnelly
        st.session_state.quant = quant
        st.success(f"✅ Found: BK:{len(detected)} · PA:{len(brooks)} · SS:{len(streets)} · DN:{len(donnelly)} · Q:{len(quant)}")
    except Exception as e:
        st.error(f"Detection error: {e}")
    finally:
        progress.empty()


# ─────────────────────────────────────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📊 Chart")
fig = render_plotly_chart(df,
    st.session_state.detected, st.session_state.brooks,
    st.session_state.streets, st.session_state.donnelly,
    st.session_state.quant, height=480)
if fig:
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['select2d','lasso2d','autoScale2d'],
    })


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS — TABBED BY ENGINE
# ─────────────────────────────────────────────────────────────────────────────
n_bk = len(st.session_state.detected)
n_pa = len(st.session_state.brooks)
n_ss = len(st.session_state.streets)
n_dn = len(st.session_state.donnelly)
n_q  = len(st.session_state.quant)

results_tabs = st.tabs([
    f"📚 BK ({n_bk})",
    f"📐 PA ({n_pa})",
    f"🎯 SS ({n_ss})",
    f"⚡ DN ({n_dn})",
    f"📈 Q ({n_q})",
])

with results_tabs[0]:
    if n_bk == 0:
        st.info("No Bulkowski patterns detected. Click RUN DETECTION above.")
    for p in st.session_state.detected:
        render_bulkowski_card(p)

with results_tabs[1]:
    if n_pa == 0:
        st.info("No Brooks PA signals detected.")
    for s in st.session_state.brooks:
        render_signal_card(s, 'brooks')

with results_tabs[2]:
    if n_ss == 0:
        st.info("No Street Smarts signals detected.")
    for s in st.session_state.streets:
        render_signal_card(s, 'streets')

with results_tabs[3]:
    if n_dn == 0:
        st.info("No Donnelly signals detected.")
    for s in st.session_state.donnelly:
        render_signal_card(s, 'donnelly')

with results_tabs[4]:
    if n_q == 0:
        st.info("No quant signals — need at least 20 bars.")
    for s in st.session_state.quant:
        with st.expander(f"{s.get('icon','◆')} **{s.get('name','Quant')}** — {s.get('signal','')}", expanded=False):
            if s.get('source'):     st.caption(f"📖 {s['source']}")
            if s.get('value'):      st.markdown(f"**Value:** {s['value']}")
            if s.get('desc'):       st.markdown(s['desc'])
            if s.get('trade_rule'): st.code(s['trade_rule'])


# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED TOOLS — Backtesting + Forecasting
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🛠️ Advanced Tools")

adv_tabs = st.tabs(["📉 Backtest", "🔮 Forecast", "📚 Library"])

# ── BACKTEST ──────────────────────────────────────────────────────────────────
with adv_tabs[0]:
    st.markdown("**Historical pattern backtest** — scans full data for past occurrences")
    if len(df) < 40:
        st.warning("Need at least 40 bars for backtesting")
    else:
        # Pattern type from dropdown — extract unique pattern types from detector
        all_pattern_types = sorted(set(engine.PATTERNS_DB.keys()))

        col_a, col_b = st.columns(2)
        with col_a:
            bt_pattern = st.selectbox("Pattern Type", all_pattern_types, key='bt_pat')
        with col_b:
            bt_dir = st.selectbox("Direction", ["BULLISH", "BEARISH"], key='bt_dir')

        col_c, col_d = st.columns(2)
        with col_c:
            bt_conf = st.slider("Min Confidence %", 30, 90, 60, key='bt_conf')
        with col_d:
            bt_hold = st.slider("Max Hold Days", 5, 60, 30, key='bt_hold')

        if st.button("📉 RUN BACKTEST", use_container_width=True):
            bt_progress = st.progress(0, text="Backtesting...")
            def cb(pct):
                bt_progress.progress(int(pct), text=f"Scanning... {pct:.0f}%")
            try:
                trades = engine.run_backtest(df, bt_pattern, bt_dir,
                    confidence_threshold=bt_conf,
                    max_hold_days=bt_hold,
                    progress_cb=cb)
                bt_progress.progress(100, text="Computing stats...")
                stats = engine.compute_backtest_stats(trades)
                bt_progress.empty()

                st.markdown(f"#### Results: {len(trades)} trades")
                if not trades:
                    st.warning("No occurrences of this pattern found in the data.")
                else:
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Win Rate", f"{stats.get('win_rate',0):.0%}")
                    sc2.metric("Avg Win", f"{stats.get('avg_win_pct',0):+.1f}%")
                    sc3.metric("Avg Loss", f"{stats.get('avg_loss_pct',0):+.1f}%")

                    sc4, sc5, sc6 = st.columns(3)
                    sc4.metric("Total P&L", f"{stats.get('total_pnl_pct',0):+.1f}%")
                    sc5.metric("Expectancy", f"{stats.get('expectancy_pct',0):+.2f}%")
                    sc6.metric("Profit Factor", f"{stats.get('profit_factor',0):.2f}")

                    # Trade table (compact for mobile)
                    if trades:
                        st.markdown("**📋 Trade Log**")
                        trade_df = pd.DataFrame([{
                            'Entry': t.get('entry_date',''),
                            'Exit':  t.get('exit_date',''),
                            'Outcome': t.get('outcome',''),
                            'P&L %':  f"{t.get('pnl_pct',0):+.1f}%",
                        } for t in trades])
                        st.dataframe(trade_df, use_container_width=True, height=300)
            except Exception as e:
                bt_progress.empty()
                st.error(f"Backtest error: {e}")

# ── FORECAST ──────────────────────────────────────────────────────────────────
with adv_tabs[1]:
    st.markdown("**Pattern completion forecast** — Bulkowski's probabilistic outcome predictor")
    if not st.session_state.detected:
        st.info("Run detection first, then come back here.")
    else:
        names = [p.get('name','?') for p in st.session_state.detected]
        sel_idx = st.selectbox("Pattern to forecast",
            options=list(range(len(names))),
            format_func=lambda i: names[i])
        if st.button("🔮 COMPUTE FORECAST", use_container_width=True):
            try:
                p = st.session_state.detected[sel_idx]
                fc = engine.compute_pattern_forecast(p, df, market_context='bull')

                st.markdown(f"#### {p.get('name','Pattern')}")
                if fc.get('verdict_tag'):
                    verdict_color = {'STRONG': '🟢', 'GOOD': '🟢', 'FAIR': '🟡',
                                     'WEAK': '🔴', 'AVOID': '🔴'}.get(fc['verdict_tag'][:6].upper(), '🟡')
                    st.markdown(f"### {verdict_color} **{fc.get('verdict','')}**")
                    if fc.get('verdict_detail'):
                        st.caption(fc['verdict_detail'])

                col_p1, col_p2, col_p3 = st.columns(3)
                if 'completion_prob' in fc:
                    col_p1.metric("Completion Prob", f"{fc['completion_prob']:.0%}")
                if 'expected_move_pct' in fc:
                    col_p2.metric("Expected Move", f"{fc['expected_move_pct']:+.1f}%")
                if 'days_to_completion' in fc:
                    col_p3.metric("Days to Target", f"{fc['days_to_completion']}")

                col_p4, col_p5, col_p6 = st.columns(3)
                if 'target_price' in fc:
                    col_p4.metric("Target", f"{fc['target_price']:.2f}")
                if 'throwback_prob' in fc:
                    col_p5.metric("Throwback Prob", f"{fc['throwback_prob']:.0%}")
                if 'failure_risk' in fc:
                    col_p6.metric("Failure Risk", fc['failure_risk'])

                if fc.get('best_conditions'):
                    st.markdown("**✅ BEST CONDITIONS**")
                    for c in fc['best_conditions']:
                        st.markdown(f"- {c}")

                if fc.get('failure_signals'):
                    st.markdown("**⚠️ FAILURE SIGNALS TO WATCH**")
                    for f in fc['failure_signals']:
                        st.markdown(f"- {f}")

                if fc.get('desc'):
                    st.markdown("---")
                    st.caption(fc['desc'])
            except Exception as e:
                st.error(f"Forecast error: {e}")

# ── LIBRARY ──────────────────────────────────────────────────────────────────
with adv_tabs[2]:
    st.markdown("**Reference Library** — full strategy database")
    lib_choice = st.radio(
        "Source", ["Bulkowski Patterns", "Street Smarts", "Donnelly"],
        horizontal=True, key='lib_choice')

    if lib_choice == "Bulkowski Patterns":
        names = sorted(engine.PATTERNS_DB.keys())
        sel = st.selectbox("Pattern", names, key='lib_bk')
        if sel:
            p = engine.PATTERNS_DB[sel]
            st.markdown(f"### {sel}")
            st.caption(f"Type: {p.get('type','?').upper()} · Direction: {p.get('direction','?').upper()}")
            if p.get('description'):
                st.markdown(p['description'])
            if p.get('identification'):
                st.markdown("**🔍 Identification:**")
                for r in p['identification']:
                    st.markdown(f"- {r}")
            if p.get('measure_rule'):
                st.info(f"📏 **Measure Rule:** {p['measure_rule']}")
            if p.get('trading_plan'):
                st.markdown("**📋 Trading Plan:**")
                tp = p['trading_plan']
                if tp.get('entry'):  st.success(f"Entry: {tp['entry']}")
                if tp.get('stop'):   st.error(f"Stop: {tp['stop']}")
                if tp.get('target'): st.warning(f"Target: {tp['target']}")
            if p.get('stats'):
                st.markdown("**📊 Stats:**")
                for k, v in p['stats'].items():
                    st.markdown(f"*{k}:*  " + ", ".join(f"{kk}={vv}" for kk, vv in v.items()))

    elif lib_choice == "Street Smarts":
        names = sorted(engine.STREET_SMARTS_DB.keys())
        sel = st.selectbox("Strategy", names, key='lib_ss')
        if sel:
            db = engine.STREET_SMARTS_DB[sel]
            st.markdown(f"### {sel}")
            st.caption(f"{db.get('source','')} · {db.get('rating','')}")
            st.markdown(db.get('concept', ''))
            if db.get('pre_market'):
                st.markdown("**📅 Pre-market checklist:**")
                for it in db['pre_market']: st.markdown(f"- {it}")
            tp = db.get('trading_plan', {})
            if tp:
                st.markdown("**📋 Trading Plan:**")
                for k in ['entry','stop','target_1','target_2','manage','avoid']:
                    if k in tp:
                        label = k.replace('_',' ').title()
                        if k == 'entry': st.success(f"**{label}:** {tp[k]}")
                        elif k == 'stop': st.error(f"**{label}:** {tp[k]}")
                        elif 'target' in k: st.warning(f"**{label}:** {tp[k]}")
                        elif k == 'avoid': st.error(f"**{label}:** {tp[k]}")
                        else: st.info(f"**{label}:** {tp[k]}")
            if db.get('review_2025'):
                st.success(f"**2025 Review:** {db['review_2025']}")

    else:  # Donnelly
        names = sorted(engine.DONNELLY_DB.keys())
        sel = st.selectbox("Setup", names, key='lib_dn')
        if sel:
            db = engine.DONNELLY_DB[sel]
            st.markdown(f"### {sel}")
            st.caption(f"{db.get('source','')} · {db.get('rating','')}")
            st.markdown(db.get('concept', ''))
            if db.get('pre_market'):
                st.markdown("**📅 Pre-market checklist:**")
                for it in db['pre_market']: st.markdown(f"- {it}")
            tp = db.get('trading_plan', {})
            if tp:
                st.markdown("**📋 Trading Plan:**")
                for k in ['entry','stop','target_1','target_2','manage','avoid']:
                    if k in tp:
                        label = k.replace('_',' ').title()
                        if k == 'entry': st.success(f"**{label}:** {tp[k]}")
                        elif k == 'stop': st.error(f"**{label}:** {tp[k]}")
                        elif 'target' in k: st.warning(f"**{label}:** {tp[k]}")
                        elif k == 'avoid': st.error(f"**{label}:** {tp[k]}")
                        else: st.info(f"**{label}:** {tp[k]}")
            if db.get('review_2025'):
                st.success(f"**2025 Review:** {db['review_2025']}")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built on the Bulkowski + Donnelly engine · Mobile UI by Streamlit · "
           "All signals computed from OHLC(V) data. Educational use only.")
