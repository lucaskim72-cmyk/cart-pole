# -*- coding: utf-8 -*-
"""Streamlit UI for the CartPole epsilon / collision simulation."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from simulation import run_simulation

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "#0d1511"
plt.rcParams["axes.facecolor"] = "#0d1511"
plt.rcParams["savefig.facecolor"] = "#0d1511"
plt.rcParams["text.color"] = "#edf1eb"
plt.rcParams["axes.labelcolor"] = "#a0ada4"
plt.rcParams["xtick.color"] = "#a0ada4"
plt.rcParams["ytick.color"] = "#a0ada4"
plt.rcParams["axes.edgecolor"] = "#25332a"

THETA_LIMIT = 12 * np.pi / 180


st.set_page_config(
    page_title="CartPole 탐험·충돌 시뮬레이터",
    page_icon="🛒",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #080d0b;
        --panel: #0d1511;
        --line: #25332a;
        --text: #edf1eb;
        --muted: #a0ada4;
        --accent: #b5f7bc;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--bg) !important;
        color: var(--text);
    }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp li {
        font-family: Pretendard, Inter, sans-serif !important;
    }
    header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"] {
        display: none !important;
    }
    .block-container {
        max-width: 1180px;
        padding: 28px 36px 72px !important;
    }
    .brand-rule {
        border-bottom: 1px solid rgba(255,255,255,0.05);
        margin: 6px 0 8px;
    }
    .wordmark {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 17px;
        font-weight: 650;
        letter-spacing: 0.04em;
    }
    .mark {
        width: 28px;
        height: 28px;
        border: 1.5px solid var(--accent);
        border-radius: 50%;
        position: relative;
    }
    .mark:after {
        content: "";
        position: absolute;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent);
        top: 9px;
        left: 9px;
    }
    .wordmark b { color: var(--accent); font-weight: 450; }
    .eyebrow {
        display: flex;
        align-items: center;
        gap: 9px;
        margin: 36px 0 18px;
        color: var(--accent);
        font-family: Consolas, "Cascadia Mono", monospace !important;
        font-size: 11px;
        letter-spacing: 0.14em;
    }
    .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 0 4px rgba(181,247,188,0.12);
    }
    .intro h1 {
        font-size: clamp(36px, 5vw, 64px) !important;
        font-weight: 450 !important;
        letter-spacing: -0.055em !important;
        line-height: 1.12 !important;
        margin: 0 0 18px !important;
        color: var(--text) !important;
    }
    .intro em { color: var(--accent); font-style: normal; }
    .intro p.lead {
        color: var(--muted);
        font-size: 15px;
        max-width: 640px;
        line-height: 1.7;
        margin-bottom: 8px;
    }
    .section-label {
        margin: 42px 0 14px;
        color: var(--muted);
        font-family: Consolas, "Cascadia Mono", monospace !important;
        font-size: 11px;
        letter-spacing: 0.14em;
    }
    .specs {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 18px;
        margin: 8px 0 6px;
    }
    .specs div {
        border-left: 1px solid var(--line);
        padding-left: 16px;
    }
    .specs b {
        display: block;
        color: var(--accent);
        font-size: 28px;
        font-weight: 450;
        letter-spacing: -0.04em;
    }
    .specs span {
        color: var(--muted);
        font-size: 11px;
        letter-spacing: 0.04em;
    }
    .metrics {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 18px;
        margin: 28px 0 8px;
    }
    .metrics article {
        border-top: 1px solid var(--line);
        padding-top: 14px;
    }
    .metrics b {
        display: block;
        color: var(--text);
        font-size: 32px;
        font-weight: 450;
        letter-spacing: -0.05em;
    }
    .metrics p {
        color: var(--muted);
        font-size: 12px;
        margin: 4px 0 0;
    }
    .chips { display: flex; gap: 8px; flex-wrap: wrap; margin: 14px 0 8px; }
    .chip {
        border: 1px solid var(--line);
        border-radius: 3px;
        color: var(--accent);
        font-family: Consolas, monospace !important;
        font-size: 10px;
        letter-spacing: 0.08em;
        padding: 5px 9px;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] div { }
    [data-testid="stWidgetLabel"] p {
        color: var(--muted) !important;
        font-size: 12px !important;
    }
    [data-baseweb="input"] {
        background: var(--bg) !important;
        border-color: var(--line) !important;
    }
    [data-testid="stImage"] img {
        border: 1px solid var(--line);
        border-radius: 7px;
        background: #fff;
    }
    [data-testid="stAlert"] {
        background: var(--panel);
        border: 1px solid var(--line);
        color: var(--muted);
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 7px; }
    .footnote {
        margin-top: 48px;
        padding-top: 18px;
        border-top: 1px solid rgba(255,255,255,0.05);
        display: flex;
        justify-content: space-between;
        color: #819389;
        font-size: 11px;
    }
    @media (max-width: 900px) {
        .metrics, .specs { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .block-container { padding: 20px 16px 48px !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_simulation(epsilon: float, collision_prob: float, steps: int, seed: int):
    return run_simulation(epsilon, collision_prob, steps, seed)


def _finish_chart(ax: plt.Axes) -> None:
    ax.set_facecolor("#0d1511")
    ax.grid(True, color="#25332a", linestyle="--", linewidth=0.6)
    legend = ax.legend(loc="upper right", fontsize=8, frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor("#0d1511")
    frame.set_edgecolor("#25332a")
    for text in legend.get_texts():
        text.set_color("#edf1eb")


def render_action_chart(result: dict) -> plt.Figure:
    steps = np.arange(1, result["metrics"]["steps"] + 1)
    actions = result["actions"]
    explored = result["explored"]

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.scatter(
        steps[~explored],
        actions[~explored],
        c="#b5f7bc",
        s=28,
        marker="o",
        alpha=0.85,
        label="이용 (각도 규칙)",
    )
    ax.scatter(
        steps[explored],
        actions[explored],
        c="#e7c27a",
        s=42,
        marker="*",
        alpha=0.9,
        label="탐험 (무작위)",
    )
    ax.set_title("행동 선택: 탐험 vs 이용")
    ax.set_xlabel("스텝")
    ax.set_ylabel("행동")
    ax.set_yticks([0, 1], ["0 왼쪽", "1 오른쪽"])
    ax.set_ylim(-0.4, 1.4)
    _finish_chart(ax)
    fig.patch.set_facecolor("#0d1511")
    fig.tight_layout()
    return fig


def render_angle_chart(result: dict) -> plt.Figure:
    steps = np.arange(1, result["metrics"]["steps"] + 1)
    angles = result["angles"]
    levels = np.asarray(result["collision_levels"], dtype=object)
    fell = result["fell"]

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(steps, angles, color="#edf1eb", linewidth=1.4, label="막대 각도")
    ax.axhline(THETA_LIMIT, color="#ff8b7a", linestyle="--", linewidth=1, label="쓰러짐 기준 (±12°)")
    ax.axhline(-THETA_LIMIT, color="#ff8b7a", linestyle="--", linewidth=1)
    ax.axhline(0, color="#25332a", linewidth=0.8)

    style = {
        "약": ("#7ee0c8", 36),
        "중": ("#e7c27a", 52),
        "강": ("#ff8b7a", 70),
    }
    for level, (color, size) in style.items():
        mask = levels == level
        if np.any(mask):
            ax.scatter(steps[mask], angles[mask], c=color, s=size, zorder=3, label=f"충돌 {level}")

    if np.any(fell):
        ax.scatter(
            steps[fell],
            angles[fell],
            facecolors="none",
            edgecolors="#edf1eb",
            s=80,
            linewidths=1.2,
            zorder=4,
            label="쓰러짐",
        )

    ax.set_title("막대 각도와 충돌 시점")
    ax.set_xlabel("스텝")
    ax.set_ylabel("각도 (rad)")
    _finish_chart(ax)
    ax.legend(loc="upper right", ncol=2, fontsize=8, frameon=True)
    legend = ax.get_legend()
    legend.get_frame().set_facecolor("#0d1511")
    legend.get_frame().set_edgecolor("#25332a")
    for text in legend.get_texts():
        text.set_color("#edf1eb")
    fig.patch.set_facecolor("#0d1511")
    fig.tight_layout()
    return fig


def collision_events(result: dict) -> list[dict[str, object]]:
    return [
        {"스텝": index + 1, "대상": target, "강도": level, "들어온 쪽": side}
        for index, (level, side, target) in enumerate(
            zip(result["collision_levels"], result["collision_sides"], result["collision_targets"])
        )
        if level is not None
    ]


def main() -> None:
    st.markdown(
        """
        <div class="wordmark"><span class="mark"></span><span>CART <b>POLE</b></span></div>
        <div class="brand-rule"></div>
        <div class="intro">
          <div class="eyebrow"><span class="dot"></span>EPSILON · COLLISION LAB</div>
          <h1>탐험 확률과 충돌을<br><em>직접 움직여</em> 봅니다</h1>
          <p class="lead">Epsilon은 무작위 탐험 비율이고, 충돌 확률은 매 스텝 옆에서 충격이 들어올 확률입니다. 이용 행동은 막대가 기울어진 반대쪽으로 카트를 밉니다.</p>
        </div>
        <div class="section-label">CONTROLS</div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    epsilon = c1.slider("Epsilon · 탐험", 0.0, 1.0, 0.30, 0.01)
    collision_prob = c2.slider("충돌 확률", 0.0, 1.0, 0.08, 0.01)
    steps = c3.slider("스텝", 50, 300, 150, 10)
    seed = c4.number_input("난수 시드", min_value=0, max_value=1_000_000, value=42, step=1)

    st.markdown(
        f"""
        <div class="specs">
          <div><b>{epsilon:.0%}</b><span>탐험</span></div>
          <div><b>{1 - epsilon:.0%}</b><span>이용</span></div>
          <div><b>{collision_prob:.0%}</b><span>스텝당 충돌</span></div>
          <div><b>{steps * collision_prob:.1f}</b><span>기대 충돌 횟수</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("시뮬레이션을 계산하고 영상을 만드는 중입니다..."):
        result = cached_simulation(float(epsilon), float(collision_prob), int(steps), int(seed))

    metrics = result["metrics"]
    events = collision_events(result)

    st.markdown(
        f"""
        <div class="metrics">
          <article><b>{metrics['falls']}</b><p>쓰러진 횟수</p></article>
          <article><b>{metrics['explore_rate']:.0%}</b><p>실제 탐험 비율 · 설정 {epsilon:.0%}</p></article>
          <article><b>{metrics['collisions']}</b><p>충돌 · 기대 {steps * collision_prob:.1f}</p></article>
          <article><b>{metrics['mean_survival']:.0f}</b><p>평균 생존 스텝</p></article>
          <article><b>{metrics['total_reward']:.0f}</b><p>총 보상</p></article>
        </div>
        <div class="chips">
          <span class="chip">약 {metrics['collision_weak']}</span>
          <span class="chip">중 {metrics['collision_mid']}</span>
          <span class="chip">강 {metrics['collision_strong']}</span>
          <span class="chip">탐험 {metrics['explore_count']} · 이용 {metrics['exploit_count']}</span>
        </div>
        <div class="section-label">SIMULATION</div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 0.85])
    with left:
        st.image(result["gif"], use_container_width=True)
        st.markdown(
            '<p class="lead" style="color:#a0ada4;font-size:13px;margin-top:10px;">옆 카트는 레일을 따라 본체를 들이받고, 금속 추는 막대를 칩니다. 범퍼 색은 강도입니다. 청록 약, 주황 중, 빨강 강.</p>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="section-label" style="margin-top:0;">LOG</div>', unsafe_allow_html=True)
        if events:
            st.dataframe(events, use_container_width=True, hide_index=True, height=280)
        else:
            st.info("이 시드와 충돌 확률에서는 충돌이 발생하지 않았습니다.")

    st.markdown('<div class="section-label">TRACES</div>', unsafe_allow_html=True)
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.pyplot(render_action_chart(result), clear_figure=True)
    with chart_right:
        st.pyplot(render_angle_chart(result), clear_figure=True)
    plt.close("all")

    st.markdown(
        """
        <div class="footnote">
          <span>CART POLE LAB</span>
          <span>EPSILON GREEDY · SIDE IMPACT</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
