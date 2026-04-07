import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Amazon Recommender — Group 8",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── FULL CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main background */
.main .block-container { 
    padding: 1.5rem 2rem 2rem;
    max-width: 1400px;
}

/* ── TOP HERO BAR ── */
.hero {
    background: linear-gradient(135deg, #0D1B2A 0%, #1E2761 50%, #2E86AB 100%);
    border-radius: 18px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; top: -40%; right: -10%;
    width: 450px; height: 450px;
    background: radial-gradient(circle, rgba(46,134,171,0.25) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title { color: white; font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; }
.hero-title span { color: #64C8E8; }
.hero-sub { color: #93C5D8; font-size: 0.95rem; margin: 0.3rem 0 0; }
.hero-team { color: #C8D8F0; font-size: 0.85rem; margin-top: 0.6rem; opacity: 0.85; }

/* ── KPI CARDS ── */
.kpi-row { display: flex; gap: 1rem; margin-bottom: 1.8rem; flex-wrap: wrap; }
.kpi {
    flex: 1; min-width: 140px;
    background: white;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    border: 1px solid #E8EFF5;
    box-shadow: 0 2px 12px rgba(13,27,42,0.06);
    position: relative;
    overflow: hidden;
}
.kpi::after {
    content: "";
    position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
    background: var(--accent, #2E86AB);
}
.kpi .num { font-size: 1.75rem; font-weight: 800; color: #0D1B2A; line-height: 1; }
.kpi .lbl { font-size: 0.72rem; color: #6B7A8D; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.3rem; }
.kpi .icon { font-size: 1.4rem; margin-bottom: 0.4rem; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #0D1B2A;
}
section[data-testid="stSidebar"] * { color: #C8D8F0 !important; }
section[data-testid="stSidebar"] .info-pill { background: #1E3A5F !important; color: #E8F4FD !important; border-left: 4px solid #FF9900 !important; }
section[data-testid="stSidebar"] .info-pill b { color: #FF9900 !important; }
section[data-testid="stSidebar"] .info-pill code { background: #0D1B2A !important; color: #93C5FD !important; }
section[data-testid="stSidebar"] .stRadio label { color: #C8D8F0 !important; }
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: white !important; }
section[data-testid="stSidebar"] .stTextInput input { 
    background: #1E2D3D !important; 
    color: white !important; 
    border: 1px solid #2E86AB !important;
    border-radius: 8px !important;
}

/* ── MODEL PILL BADGES ── */
.model-pop  { background: #064E3B; color: #6EE7B7; border-radius: 999px; padding: .2rem .8rem; font-size:.78rem; font-weight:600; }
.model-item { background: #1E3A5F; color: #93C5FD; border-radius: 999px; padding: .2rem .8rem; font-size:.78rem; font-weight:600; }
.model-user { background: #3B1F5E; color: #C4B5FD; border-radius: 999px; padding: .2rem .8rem; font-size:.78rem; font-weight:600; }

/* ── RESULT CARDS ── */
.result-section {
    background: white; border-radius: 16px; padding: 1.5rem;
    border: 1px solid #E8EFF5;
    box-shadow: 0 4px 20px rgba(13,27,42,0.07);
    margin-bottom: 1.2rem;
}
.section-label {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #6B7A8D; margin-bottom: 1rem;
}

/* ── RATED ITEM ── */
.rated-item {
    display: flex; align-items: center; gap: .75rem;
    padding: .6rem .8rem; border-radius: 10px;
    background: #F0FDF4; border-left: 3px solid #10B981;
    margin-bottom: .45rem;
}
.rated-stars { font-size: .85rem; }
.rated-name  { font-size: .88rem; font-weight: 500; color: #0D1B2A; }

/* ── RECOMMENDATION CARD ── */
.rec-card {
    display: flex; align-items: center; gap: 1rem;
    padding: .85rem 1rem; border-radius: 12px;
    background: white; border: 1px solid #E8EFF5;
    box-shadow: 0 2px 8px rgba(13,27,42,0.05);
    margin-bottom: .55rem;
    transition: transform 0.15s, box-shadow 0.15s;
}
.rec-rank {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: .95rem; color: white;
    flex-shrink: 0;
}
.rec-name  { font-size: .92rem; font-weight: 600; color: #0D1B2A; }
.rec-score { font-size: .78rem; color: #6B7A8D; margin-top: .15rem; }
.rec-id    { font-size: .72rem; color: #94A3B8; }

/* ── FORMULA BOX ── */
.formula-box {
    background: #0D1B2A; border-radius: 10px;
    padding: .85rem 1.1rem; font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: .88rem; color: #93C5FD; margin: .6rem 0;
}

/* ── INFO PILL ── */
.info-pill {
    background: #EFF6FF; border-radius: 10px; padding: .9rem 1.1rem;
    border-left: 4px solid #2E86AB; font-size: .88rem; color: #1E3A5F;
    margin: .5rem 0; line-height: 1.6;
}

/* ── EMPTY STATE ── */
.empty-state {
    text-align: center; padding: 4rem 2rem; color: #94A3B8;
}
.empty-state .icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-state .msg  { font-size: 1.05rem; color: #64748B; }

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #1E2761, #2E86AB) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: .95rem !important; padding: .6rem 1.5rem !important;
    width: 100% !important; transition: opacity 0.2s !important;
    box-shadow: 0 4px 12px rgba(46,134,171,0.35) !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── COMPARISON TABLE ── */
.cmp-table { width: 100%; border-collapse: collapse; font-size: .88rem; }
.cmp-table th { background: #0D1B2A; color: white; padding: .7rem 1rem; text-align: left; }
.cmp-table th:first-child { border-radius: 8px 0 0 0; }
.cmp-table th:last-child  { border-radius: 0 8px 0 0; }
.cmp-table td { padding: .6rem 1rem; border-bottom: 1px solid #E8EFF5; }
.cmp-table tr:nth-child(even) td { background: #F8FAFC; }
.cmp-best { background: #E8F5E9 !important; font-weight: 600; color: #065F46; }
.tick { color: #10B981; font-weight: 700; }
.cross { color: #EF4444; font-weight: 700; }
.warn  { color: #F59E0B; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset and computing models...")
def load_all():
    df = pd.read_excel("Group8-2.xlsx")
    uc = df.groupby("UserId")["Rating"].count()
    pc = df.groupby("ProductId")["Rating"].count()
    df_cf = df[df["UserId"].isin(uc[uc>=3].index) & df["ProductId"].isin(pc[pc>=3].index)].copy()
    rm = df_cf.pivot_table(index="UserId", columns="ProductId", values="Rating").fillna(0)
    i_sim = cosine_similarity(rm.T)
    u_sim = cosine_similarity(rm)
    idf = pd.DataFrame(i_sim, index=rm.columns, columns=rm.columns)
    udf = pd.DataFrame(u_sim, index=rm.index,   columns=rm.index)
    return df, df_cf, rm, idf, udf

df, df_cf, rm, idf, udf = load_all()
sparsity = 1 - len(df) / (df["UserId"].nunique() * df["ProductId"].nunique())


# ── MODEL FUNCTIONS ───────────────────────────────────────────────────────────
def popularity_rec(dataframe, min_ratings=5, n=5):
    stats = dataframe.groupby(["ProductId","product_name"])["Rating"].agg(
        avg_rating="mean", num_ratings="count").reset_index()
    C = stats["avg_rating"].mean(); m = min_ratings
    stats["score"] = (stats["num_ratings"]/(stats["num_ratings"]+m))*stats["avg_rating"] + \
                     (m/(stats["num_ratings"]+m))*C
    result = stats[stats["num_ratings"]>=min_ratings].sort_values("score",ascending=False).head(n)[
        ["product_name","avg_rating","num_ratings","score"]].reset_index(drop=True)
    result.index = result.index + 1
    return result

def item_rec(uid, rm, idf, lookup, n=5):
    if uid not in rm.index: return None
    ur = rm.loc[uid]; rated = ur[ur>0].index.tolist()
    if not rated: return None
    unrated = ur[ur==0].index
    scores = {p: idf.loc[p, rated].mean() for p in unrated}
    top = pd.Series(scores).sort_values(ascending=False).head(n).reset_index()
    top.columns = ["ProductId","score"]
    i2n = lookup.drop_duplicates("ProductId").set_index("ProductId")["product_name"]
    top["product_name"] = top["ProductId"].map(i2n)
    top.index = top.index + 1
    return top

def user_rec(uid, rm, udf, lookup, k=10, n=5):
    if uid not in rm.index: return None
    ur = rm.loc[uid]; rated = ur[ur>0].index
    neighbors = udf[uid].drop(uid).sort_values(ascending=False).head(k).index
    pred = rm.loc[neighbors].mean(axis=0).drop(rated).sort_values(ascending=False).head(n).reset_index()
    pred.columns = ["ProductId","score"]
    i2n = lookup.drop_duplicates("ProductId").set_index("ProductId")["product_name"]
    pred["product_name"] = pred["ProductId"].map(i2n)
    pred.index = pred.index + 1
    return pred


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    model = st.radio("**Choose a model:**", [
        "🏆 Popularity-Based",
        "🔗 Item-Based CF",
        "👥 User-Based CF"
    ])

    st.markdown("---")

    if "Pop" not in model:
        uid_input = st.text_input("**User ID:**", placeholder="e.g. A2XQ4JEGR4YT3B")
        if st.button("🎲 Random User"):
            st.session_state["rnd"] = str(np.random.choice(rm.index.tolist()))
        if "rnd" in st.session_state:
            uid_input = st.session_state["rnd"]
            st.code(uid_input, language=None)
    else:
        uid_input = ""
        st.info("Popularity model shows the same top-5 for all users.")

    st.markdown("---")
    go = st.button("🚀 Get Recommendations")

    # Model info
    st.markdown("---")
    if "Pop" in model:
        st.markdown("""<div class="info-pill">
        <b>Popularity-Based</b><br>
        Same top-5 for everyone. Bayesian weighted score prevents fake-review inflation.<br><br>
        <code>score = n/(n+m) × avg + m/(n+m) × C</code>
        </div>""", unsafe_allow_html=True)
    elif "Item" in model:
        st.markdown("""<div class="info-pill">
        <b>Item-Based CF</b><br>
        Finds products with similar rating patterns to what you already liked.<br><br>
        <code>sim(X,Y) = X·Y / (‖X‖ × ‖Y‖)</code>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="info-pill">
        <b>User-Based CF</b><br>
        Finds 10 users most similar to you, recommends what they loved.<br><br>
        <code>k=10 neighbours, .drop(user_id) excludes self</code>
        </div>""", unsafe_allow_html=True)


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg"
       style="height:38px;margin-bottom:1rem;filter:brightness(0) invert(1);display:block;margin-left:auto;margin-right:auto;" alt="Amazon"/>
  <div class="hero-title">Product <span>Recommendation System</span></div>
  <div class="hero-sub">Popularity-Based &nbsp;·&nbsp; Item-Based CF &nbsp;·&nbsp; User-Based CF</div>
  <div class="hero-team">Group 8 — TBS Education &nbsp;|&nbsp; EL MIR Otmane &nbsp;•&nbsp; SAVOYE Raphael &nbsp;•&nbsp; MOUMNI Youssef</div>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
kpi_data = [
    ("📦", f"{len(df):,}", "Total Reviews", "#2E86AB"),
    ("👤", f"{df['UserId'].nunique():,}", "Unique Users", "#0D6E4E"),
    ("🏷️", f"{df['ProductId'].nunique():,}", "Products", "#7C3AED"),
    ("⭐", f"{df['Rating'].mean():.2f}", "Avg Rating", "#D97706"),
    ("📊", f"{sparsity:.2%}", "Sparsity", "#DC2626"),
]
for col, (icon, num, lbl, color) in zip([c1,c2,c3,c4,c5], kpi_data):
    col.markdown(f"""
    <div class="kpi" style="--accent:{color};">
      <div class="icon">{icon}</div>
      <div class="num">{num}</div>
      <div class="lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── MAIN CONTENT ──────────────────────────────────────────────────────────────

if go or ("rnd" in st.session_state and "Pop" not in model):

    # ── POPULARITY ────────────────────────────────────────────────────────────
    if "Pop" in model:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🏆 Popularity-Based — Top 5 Products (same for all users)</div>', unsafe_allow_html=True)
        recs = popularity_rec(df)
        colors = ["#064E3B","#065F46","#047857","#059669","#10B981"]
        for idx, row in recs.iterrows():
            stars = "⭐" * round(row["avg_rating"])
            st.markdown(f"""
            <div class="rec-card">
              <div class="rec-rank" style="background:{colors[idx-1]};">{idx}</div>
              <div style="flex:1;">
                <div class="rec-name">{str(row['product_name'])[:70]}</div>
                <div class="rec-score">Bayesian Score: <b>{row['score']:.4f}</b> &nbsp;|&nbsp;
                  Avg: {row['avg_rating']:.2f}/5 &nbsp;|&nbsp; {int(row['num_ratings'])} reviews &nbsp;|&nbsp; {stars}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.info("💡 **Why Popularity?** Solves the cold-start problem — perfect for new users with no rating history. Bayesian scoring prevents products with 1 fake 5-star review from ranking first.")

    # ── CF MODELS ─────────────────────────────────────────────────────────────
    else:
        uid = uid_input.strip() if uid_input else ""
        if not uid:
            st.warning("👈 Please enter a User ID in the sidebar or click **Random User**")
        elif uid not in rm.index:
            st.error(f"❌ User **{uid}** not found in the filtered dataset. Click **Random User** to pick a valid one.")
        else:
            col_left, col_right = st.columns([1, 1], gap="large")

            with col_left:
                # Already rated
                rated_df = df_cf[df_cf["UserId"]==uid][["product_name","Rating"]].sort_values("Rating",ascending=False)
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                st.markdown(f'<div class="section-label">📦 Products already rated by this user ({len(rated_df)} total)</div>', unsafe_allow_html=True)
                for _, row in rated_df.head(6).iterrows():
                    stars = "⭐" * int(row["Rating"])
                    name  = str(row["product_name"])[:60]
                    st.markdown(f"""
                    <div class="rated-item">
                      <div class="rated-stars">{stars}</div>
                      <div class="rated-name">{name}</div>
                    </div>""", unsafe_allow_html=True)
                if len(rated_df) > 6:
                    st.caption(f"...and {len(rated_df)-6} more rated products")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_right:
                # Recommendations
                if "Item" in model:
                    recs  = item_rec(uid, rm, idf, df_cf)
                    slbl  = "Similarity score"
                    colors = ["#1E3A5F","#1E4976","#1A5276","#2471A3","#2E86AB"]
                    badge = "Item-Based CF"
                elif "User" in model:
                    recs  = user_rec(uid, rm, udf, df_cf)
                    slbl  = "Predicted rating"
                    colors = ["#3B1F5E","#4A235A","#6C3483","#7D3C98","#8E44AD"]
                    badge = "User-Based CF"

                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                st.markdown(f'<div class="section-label">🎯 Top 5 Recommendations — {badge}</div>', unsafe_allow_html=True)

                if recs is None:
                    st.warning("Not enough data for this user.")
                else:
                    for idx, row in recs.iterrows():
                        name  = str(row["product_name"])[:60] if pd.notna(row.get("product_name")) else row["ProductId"]
                        score = row["score"]
                        pid   = str(row["ProductId"])[:28]
                        st.markdown(f"""
                        <div class="rec-card">
                          <div class="rec-rank" style="background:{colors[idx-1]};">{idx}</div>
                          <div style="flex:1;">
                            <div class="rec-name">{name}</div>
                            <div class="rec-score">{slbl}: <b>{score:.4f}</b></div>
                            <div class="rec-id">{pid}</div>
                          </div>
                        </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

elif "Pop" not in model:
    st.markdown("""
    <div class="empty-state">
      <div class="icon">🛒</div>
      <div class="msg">Select a model, enter a User ID in the sidebar<br>and click <b>Get Recommendations</b></div>
    </div>""", unsafe_allow_html=True)

# ── COMPARISON TABLE ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📊 View Model Comparison Table"):
    st.markdown("""
    <table class="cmp-table">
      <thead><tr>
        <th>Criterion</th>
        <th>🏆 Popularity-Based</th>
        <th>🔗 Item-Based CF</th>
        <th>👥 User-Based CF</th>
      </tr></thead>
      <tbody>
        <tr><td><b>Personalised?</b></td><td class="cross">✗ No — same for all</td><td class="tick">✓ Yes</td><td class="tick">✓ Yes</td></tr>
        <tr><td><b>Cold-start</b></td><td class="cmp-best">✓ Works perfectly</td><td class="cross">✗ Needs history</td><td class="cross">✗ Needs history</td></tr>
        <tr><td><b>Scalability</b></td><td class="tick">✓ O(1)</td><td class="tick cmp-best">✓ Pre-computed offline</td><td class="warn">⚠ O(n²) users</td></tr>
        <tr><td><b>Sparsity robust</b></td><td class="tick">✓ Not affected</td><td class="warn">⚠ Moderate</td><td class="cross">✗ Low</td></tr>
        <tr><td><b>Serendipity</b></td><td>Low</td><td>Medium</td><td>High</td></tr>
        <tr><td><b>Production ready</b></td><td class="tick">✓ Yes</td><td class="tick cmp-best">✓ Amazon's approach</td><td class="cross">✗ Not at scale</td></tr>
      </tbody>
    </table>
    <br>
    <div class="info-pill"><b>Our Recommendation:</b> Use a hybrid strategy — Popularity for new users (0–2 ratings), Item-Based CF once the user has 3+ ratings, User-Based CF as a serendipity layer for the most active users.</div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Group 8 — TBS Education | EL MIR Otmane • SAVOYE Raphael • MOUMNI Youssef | Amazon Product Reviews | 20,000 reviews | 3 Recommendation Models")
