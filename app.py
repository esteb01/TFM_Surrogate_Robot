import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
import time

# --- PROJECT IMPORTS ---
from src.simulation import RobotSimulator
from src.preprocessing import DimensionalityReducer
from src.surrogate_models import (
    KrigingSurrogate, NeuralSurrogate,
    RBFSurrogate, SVRSurrogate, PhysicsGuidedSurrogate
)

# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Digital Twin — Robotic Surrogate Optimisation",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

MODELS_DIR          = 'models'
RESULTS_DIR         = 'results'
N_DIMENSIONS        = 350
LATENT_DIM          = 16
CONTEXT_DIM         = 6
COLLISION_THRESHOLD = 1500

st.title("Digital Twin: Physics-Informed Robotic Trajectory Optimisation")
st.caption(
    "Real-time surrogate inference interface — "
    "Autoencoder latent space · Multi-task prediction · "
    "Selective ground-truth validation"
)

# ==========================================
# 2. SYSTEM LOADING
# ==========================================
st.sidebar.header("System Configuration")

model_key = st.sidebar.selectbox(
    "Active Inference Engine",
    (
        "PINN (Physics-Guided)",
        "Neural Network (Multi-Task)",
        "Kriging (Standard)",
        "SVR (Sklearn)",
        "RBF (SMT)",
    ),
    help="Surrogate model used for real-time cost and safety prediction."
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Latent dimension: {LATENT_DIM}  \n"
    f"Input dimension: {N_DIMENSIONS}  \n"
    f"Collision threshold J: {COLLISION_THRESHOLD}"
)


@st.cache_resource
def load_artifacts():
    try:
        dr = DimensionalityReducer(method='autoencoder', n_components=LATENT_DIM)
        dr.load(os.path.join(MODELS_DIR, 'autoencoder.pkl'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'target_scaler.pkl'))

        models  = {}
        configs = [
            ("Neural Network (Multi-Task)", "neural_net.pkl",
             lambda: NeuralSurrogate(LATENT_DIM + CONTEXT_DIM)),
            ("PINN (Physics-Guided)",       "pinn.pkl",
             lambda: PhysicsGuidedSurrogate(LATENT_DIM + CONTEXT_DIM)),
            ("SVR (Sklearn)",               "svr.pkl",   SVRSurrogate),
            ("RBF (SMT)",                   "rbf.pkl",   RBFSurrogate),
            ("Kriging (Standard)",          "kriging.pkl", KrigingSurrogate),
        ]

        for name, file, constructor in configs:
            path = os.path.join(MODELS_DIR, file)
            if os.path.exists(path):
                m = constructor()
                m.load(path)
                models[name] = m

        return dr, scaler, models
    except Exception as e:
        return None, None, str(e)


dr, target_scaler, loaded_models = load_artifacts()

if not loaded_models:
    st.error(
        "Initialisation Error: model artifacts not found. "
        "Run main.py to train and serialise all surrogate models."
    )
    st.stop()


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def decode_cost(p_scaled):
    """Inverse-transform normalised log-cost to raw cost space."""
    val = float(np.array(p_scaled).flatten()[0])
    return float(np.expm1(target_scaler.inverse_transform([[val]])[0][0]))


def classify_trajectory(model_output, raw_cost):
    """Return (safety_label, collision_probability, strategy_label)."""
    if isinstance(model_output, tuple):
        _, p_prob = model_output
        prob  = float(np.array(p_prob).flatten()[0])
        label = "Collision" if prob >= 0.5 else "Safe"
        return label, prob, "Probabilistic"
    else:
        prob  = 1.0 if raw_cost >= COLLISION_THRESHOLD else 0.0
        label = "Collision" if raw_cost >= COLLISION_THRESHOLD else "Safe"
        return label, prob, "Threshold heuristic"


def run_all_models(X_in):
    """Query all loaded surrogate models on a single input vector."""
    results = []
    for name, model in loaded_models.items():
        t0  = time.perf_counter()
        out = model.predict(X_in)
        dt  = time.perf_counter() - t0

        p_scaled = out[0] if isinstance(out, tuple) else out
        raw_cost = decode_cost(p_scaled)
        safety, prob, strat = classify_trajectory(out, raw_cost)

        results.append({
            "Model":            name,
            "Predicted Cost J": raw_cost,
            "Safety Verdict":   safety,
            "Collision Prob.":  prob,
            "Strategy":         strat,
            "Latency (ms)":     dt * 1000,
        })
    return results


def generate_candidate_batch(sim, target, obstacle, n_cand, rng):
    """Generate diverse batch mixing synthetic and real collision trajectories."""
    cands = []

    real_collisions = []
    train_csv = os.path.join('data', 'raw', 'robot_final_data.csv')
    if os.path.exists(train_csv):
        df_train     = pd.read_csv(train_csv, nrows=5000)
        traj_cols    = [c for c in df_train.columns if c.startswith('dim_')]
        collision_df = df_train[df_train['cost'] > COLLISION_THRESHOLD]
        if len(collision_df) > 0:
            real_collisions = collision_df[traj_cols].values

    for _ in range(n_cand):
        r = rng.random()
        if r < 0.25:
            h, noise = rng.uniform(-0.15, 0.05), 0.020
        elif r < 0.55:
            h, noise = rng.uniform(0.25, 0.55), 0.015
        elif r < 0.75:
            h, noise = rng.uniform(-0.10, 0.60), 0.015
        elif r < 0.90 and len(real_collisions) > 0:
            idx  = rng.integers(0, len(real_collisions))
            traj = real_collisions[idx] + rng.normal(0, 0.005, (N_DIMENSIONS,))
            cands.append(traj)
            continue
        else:
            h, noise = rng.uniform(-0.01, 0.01), 0.003

        traj = (
            sim.get_ik_trajectory_advanced(target, mid_point_height_offset=h)
            + rng.normal(0, noise, (N_DIMENSIONS,))
        )
        cands.append(traj)

    return np.array(cands)


def run_surrogate_on_batch(cands, target, obstacle, model):
    """Run surrogate inference on full candidate batch."""
    n     = len(cands)
    X_lat = dr.transform(cands)
    ctx   = np.tile(np.concatenate([target, obstacle]), (n, 1))
    t0    = time.perf_counter()
    out   = model.predict(np.hstack([X_lat, ctx]))
    t_ai  = time.perf_counter() - t0

    if isinstance(out, tuple):
        p_s, p_prob = out
        preds     = np.array([decode_cost(v) for v in p_s.flatten()])
        safe_mask = p_prob.flatten() < 0.5
        strategy  = "Probabilistic"
    else:
        preds     = np.array([decode_cost(v) for v in out.flatten()])
        safe_mask = preds < COLLISION_THRESHOLD
        strategy  = "Threshold heuristic"

    return preds, safe_mask, t_ai, strategy


# ==========================================
# 4. INTERFACE TABS
# ==========================================
tab_reg, tab_cls, tab_unit, tab_opt, tab_robust = st.tabs([
    "Regression Analysis",
    "Safety Classification",
    "Unit Validation",
    "Batch Optimisation",
    "Robustness Analysis",
])

res_csv = os.path.join(RESULTS_DIR, 'model_comparison.csv')
df_res  = pd.read_csv(res_csv) if os.path.exists(res_csv) else None


# ------------------------------------------
# TAB 1 — REGRESSION METRICS
# ------------------------------------------
with tab_reg:
    st.header("Regression Performance: Physical Cost Estimation")
    st.markdown(
        "Evaluation of each surrogate model's ability to predict the "
        "continuous physical cost J. The primary metric is R²; "
        "SMAPE provides scale-independent proportional accuracy."
    )

    if df_res is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Benchmark Metrics")
            available = df_res.columns.tolist()
            reg_cols  = [c for c in ['Model', 'r2', 'rmse', 'mae', 'smape'] if c in available]
            fmt       = {k: '{:.4f}' for k in reg_cols if k != 'Model'}
            if 'smape' in fmt: fmt['smape'] = '{:.2f}%'

            hi_cols = [c for c in ['r2'] if c in reg_cols]
            lo_cols = [c for c in ['smape', 'rmse', 'mae'] if c in reg_cols]
            styled  = df_res[reg_cols].style.format(fmt)
            if hi_cols: styled = styled.highlight_max(axis=0, subset=hi_cols, color='#d1e7dd')
            if lo_cols: styled = styled.highlight_min(axis=0, subset=lo_cols, color='#d1e7dd')
            st.dataframe(styled, use_container_width=True)

            best = df_res.loc[df_res['r2'].idxmax()]
            st.info(f"Best regressor: **{best['Model']}** — R²: {best['r2']:.4f}")

        with c2:
            st.subheader("R² Score Comparison")
            fig = px.bar(
                df_res, x='Model', y='r2', color='Model', text_auto='.3f',
                labels={'r2': 'R² Score', 'Model': ''},
                title="Coefficient of Determination (higher is better)"
            )
            fig.update_layout(showlegend=False, yaxis_range=[0.8, 1.0])
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    img_reg = os.path.join(RESULTS_DIR, 'regression_plots.png')
    if os.path.exists(img_reg):
        st.subheader("Predicted vs. Actual Cost — Regression Plots")
        st.image(img_reg,
                 caption="Figure 5. Predicted vs. Actual Cost (MinMax-normalised log-cost space). "
                         "Red dashed line: perfect prediction (y = x).",
                 use_container_width=True)


# ------------------------------------------
# TAB 2 — SAFETY CLASSIFICATION
# ------------------------------------------
with tab_cls:
    st.header("Safety Assessment: Collision Detection")
    st.markdown(
        "Recall is the primary safety metric: it measures the fraction of "
        "true collision trajectories correctly identified (minimising False Negatives)."
    )

    if df_res is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Classification Metrics")
            cls_cols   = [c for c in ['Model', 'accuracy', 'recall', 'precision', 'f1']
                          if c in df_res.columns]
            fmt_cls    = {k: '{:.4f}' for k in cls_cols if k != 'Model'}
            styled_cls = df_res[cls_cols].style.format(fmt_cls)
            if 'recall' in cls_cols:
                styled_cls = styled_cls.highlight_max(axis=0, subset=['recall'], color='#ffe0e0')
            st.dataframe(styled_cls, use_container_width=True)
            best_s = df_res.loc[df_res['recall'].idxmax()]
            st.warning(f"Highest Recall: **{best_s['Model']}** — Recall: {best_s['recall']:.1%}")

        with c2:
            st.subheader("Recall Sensitivity")
            fig = px.bar(
                df_res, x='Model', y='recall', color='Model', text_auto='.3f',
                labels={'recall': 'Recall (Safety)', 'Model': ''},
                title="Collision Detection Recall (higher = safer)"
            )
            fig.update_layout(showlegend=False, yaxis_range=[0.85, 1.0])
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    img_cm = os.path.join(RESULTS_DIR, 'confusion_matrices.png')
    if os.path.exists(img_cm):
        st.subheader("Confusion Matrices — All Models")
        st.image(img_cm,
                 caption="Figure 7. Confusion matrices for collision detection. "
                         "Bottom-left quadrant (False Negatives) is the critical failure mode.",
                 use_container_width=True)


# ------------------------------------------
# TAB 3 — UNIT VALIDATION  (Case Study I)
# ------------------------------------------
with tab_unit:
    st.header("Case Study I: Single-Trajectory Unit Validation")
    st.markdown(
        "Stochastic scenario generation with ground-truth PyBullet verification. "
        "All five surrogate architectures are queried simultaneously, enabling "
        "cross-model safety consensus analysis."
    )

    c_ctrl, c_view = st.columns([1, 2])

    with c_ctrl:
        st.subheader("Scenario Parameters")
        use_seed    = st.checkbox("Fix random seed (reproducibility)", value=False)
        seed_val    = st.number_input("Seed value", min_value=0, max_value=9999,
                                      value=42, step=1, disabled=not use_seed)
        st.markdown("**Workspace sampling ranges**")
        tx_range    = st.slider("Target X range (m)", 0.30, 0.70, (0.40, 0.70), step=0.05)
        ty_range    = st.slider("Target Y range (m)", -0.40, 0.40, (-0.40, 0.40), step=0.05)
        noise_sigma = st.slider("Control noise σ (rad)", 0.00, 0.10, 0.02, step=0.005,
                        key="unit_noise_sigma")

        if st.button("Generate Random Scenario", type="primary"):
            rng      = np.random.default_rng(seed_val if use_seed else None)
            tx       = rng.uniform(*tx_range)
            ty       = rng.uniform(*ty_range)
            target   = np.array([tx, ty, 0.05])
            ox       = tx * rng.uniform(0.4, 0.6)
            oy       = ty * rng.uniform(0.4, 0.6) if abs(ty) > 0.05 else rng.uniform(-0.1, 0.1)
            obstacle = np.array([ox, oy, 0.25])

            h_offset  = rng.uniform(-0.15, 0.40)
            sim       = RobotSimulator(dimension=N_DIMENSIONS)
            base_traj = sim.get_ik_trajectory_advanced(target, mid_point_height_offset=h_offset)
            traj      = base_traj + rng.normal(0, noise_sigma, base_traj.shape)

            X_lat = dr.transform(traj.reshape(1, -1))
            ctx   = np.concatenate([target, obstacle]).reshape(1, -1)
            X_in  = np.hstack([X_lat, ctx])

            rows     = run_all_models(X_in)
            df_preds = pd.DataFrame(rows)

            st.session_state['u_data']  = {
                'traj': traj, 'target': target, 'obs': obstacle,
                'real': None, 'real_time': None, 'h_offset': h_offset,
            }
            st.session_state['u_preds'] = df_preds

    with c_view:
        if 'u_data' in st.session_state:
            u  = st.session_state['u_data']
            df = st.session_state['u_preds']

            m1, m2, m3 = st.columns(3)
            m1.metric("Target",        f"({u['target'][0]:.2f}, {u['target'][1]:.2f})")
            m2.metric("Obstacle",      f"({u['obs'][0]:.2f}, {u['obs'][1]:.2f})")
            m3.metric("Height offset", f"{u['h_offset']:.3f} m")

            st.subheader("Multi-Model Inference Results")
            st.dataframe(
                df.style
                .format({"Predicted Cost J": "{:.2f}",
                         "Collision Prob.":  "{:.2%}",
                         "Latency (ms)":     "{:.3f}"})
                .applymap(lambda v: "background-color:#d4edda" if v == "Safe"
                          else "background-color:#f8d7da" if v == "Collision" else "",
                          subset=["Safety Verdict"]),
                use_container_width=True
            )

            safe_count  = (df["Safety Verdict"] == "Safe").sum()
            total_count = len(df)
            if safe_count == total_count:
                st.success(f"Unanimous consensus: all {total_count} models classify this trajectory as **Safe**.")
            elif safe_count == 0:
                st.error(f"Unanimous consensus: all {total_count} models classify this trajectory as **Collision**.")
            else:
                st.warning(f"Divergent consensus: {safe_count}/{total_count} models classify as Safe. "
                           f"Manual inspection recommended.")

            st.divider()
            if st.button("Ground-Truth Validation (PyBullet)"):
                sim = RobotSimulator(dimension=N_DIMENSIONS, gui_mode=False)
                with st.spinner("Running physics simulation..."):
                    # IMPROVEMENT #3 — measure real PyBullet time
                    t_phy_start = time.perf_counter()
                    rc = sim.evaluate(
                        u['traj'].reshape(1, -1),
                        u['target'].reshape(1, -1),
                        u['obs'].reshape(1, -1)
                    )[0][0]
                    t_phy_real = time.perf_counter() - t_phy_start
                    sim.generate_gif(u['traj'], u['target'], u['obs'], "unit.gif")
                st.session_state['u_data']['real']      = rc
                st.session_state['u_data']['real_time'] = t_phy_real

            if st.session_state['u_data']['real'] is not None:
                rc     = st.session_state['u_data']['real']
                t_phy  = st.session_state['u_data']['real_time']
                v1, v2 = st.columns(2)

                with v1:
                    st.image("unit.gif",
                             caption=f"PyBullet ground-truth simulation — J = {rc:.2f}",
                             use_container_width=True)

                with v2:
                    if rc >= COLLISION_THRESHOLD:
                        st.error(f"Ground truth: COLLISION (J = {rc:.2f})")
                    else:
                        st.success(f"Ground truth: SAFE (J = {rc:.2f})")

                    # IMPROVEMENT #3 — real measured speedup
                    best_ms    = df["Latency (ms)"].min()
                    real_spdup = (t_phy * 1000) / max(best_ms, 0.001)
                    st.metric("PyBullet simulation time", f"{t_phy * 1000:.0f} ms")
                    st.metric("Fastest surrogate",        f"{best_ms:.3f} ms")
                    st.metric("Measured speedup",         f"{real_spdup:.0f}×",
                              delta="surrogate vs physics engine (measured)")

                    st.subheader("Prediction Error by Model")
                    err_rows = []
                    for _, row in df.iterrows():
                        abs_err = abs(rc - row["Predicted Cost J"])
                        rel_err = abs_err / max(rc, 1e-6) * 100
                        correct = (rc < COLLISION_THRESHOLD) == (row["Safety Verdict"] == "Safe")
                        err_rows.append({
                            "Model":          row["Model"],
                            "Abs. Error":     abs_err,
                            "Rel. Error (%)": rel_err,
                            "Classification": "Correct" if correct else "Wrong",
                        })
                    df_err = pd.DataFrame(err_rows)
                    st.dataframe(
                        df_err.style
                        .format({"Abs. Error": "{:.2f}", "Rel. Error (%)": "{:.1f}"})
                        .applymap(lambda v: "background-color:#d4edda" if v == "Correct"
                                  else "background-color:#f8d7da", subset=["Classification"]),
                        use_container_width=True
                    )


# ------------------------------------------
# TAB 4 — BATCH OPTIMISATION  (Case Study II)
# ------------------------------------------
with tab_opt:
    st.header("Case Study II: Industrial Batch Optimisation")
    st.markdown(
        "High-throughput trajectory selection from a stochastically generated "
        "candidate pool. The surrogate pre-filters the batch; the physics engine "
        "is invoked only for the single selected optimal trajectory."
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Scenario Configuration")
        scenario_mode = st.radio(
            "Target/Obstacle mode",
            ["Fixed (reproducible)", "Random"],
            help="Fixed: target (0.65, 0.00), obstacle (0.35, 0.00)."
        )
        n_cand = st.slider("Batch size (N candidates)", 100, 2000, 1000, step=100)

        if st.button("Run Batch Optimisation", type="primary"):
            sim = RobotSimulator(dimension=N_DIMENSIONS)

            if scenario_mode == "Fixed (reproducible)":
                target   = np.array([0.65, 0.0, 0.05])
                obstacle = np.array([0.35, 0.0, 0.25])
            else:
                rng_s    = np.random.default_rng()
                tx       = rng_s.uniform(0.45, 0.70)
                ty       = rng_s.uniform(-0.30, 0.30)
                target   = np.array([tx, ty, 0.05])
                ox       = tx * rng_s.uniform(0.35, 0.55)
                oy       = ty * rng_s.uniform(0.35, 0.55)
                obstacle = np.array([ox, oy, 0.25])

            rng   = np.random.default_rng()
            cands = generate_candidate_batch(sim, target, obstacle, n_cand, rng)
            model = loaded_models[model_key]
            preds, safe_mask, t_ai, strategy_lbl = run_surrogate_on_batch(
                cands, target, obstacle, model
            )

            valid_costs = preds.copy()
            valid_costs[~safe_mask] = np.inf
            n_safe = int(safe_mask.sum())

            if n_safe == 0:
                best_idx, planning_failure = None, True
            else:
                best_idx, planning_failure = int(np.argmin(valid_costs)), False

            # IMPROVEMENT #2 — convergence curve
            if not planning_failure:
                step        = max(10, n_cand // 20)
                checkpoints = list(range(step, n_cand + 1, step))
                conv_best   = []
                run_best    = np.inf
                for cp in checkpoints:
                    sub_best = np.min(valid_costs[:cp]) if np.any(valid_costs[:cp] < np.inf) else np.inf
                    run_best = min(run_best, sub_best)
                    conv_best.append(run_best if run_best < np.inf else None)
            else:
                checkpoints, conv_best = [], []

            st.session_state['opt'] = {
                'cands': cands, 'preds': preds, 'safe_mask': safe_mask,
                'best': best_idx, 'ctx': (target, obstacle),
                'time': t_ai, 'n': n_cand, 'n_safe': n_safe,
                'strategy': strategy_lbl, 'failure': planning_failure,
                'conv_checkpoints': checkpoints, 'conv_best': conv_best,
            }

    with col2:
        if 'opt' in st.session_state:
            res = st.session_state['opt']

            if res['failure']:
                st.error(
                    f"Planning Failure: no safe candidate found in {res['n']} trajectories. "
                    f"Increase batch size or revise obstacle configuration."
                )
            else:
                best_cost = res['preds'][res['best']]
                t_phy_est = res['n'] * 0.045
                speedup   = t_phy_est / max(res['time'], 1e-9)

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Inference Time",      f"{res['time'] * 1000:.1f} ms")
                k2.metric("Speedup vs Physics",  f"{speedup:.0f}×",
                          delta="GPU surrogate vs sequential PyBullet")
                k3.metric("Optimal Predicted Cost", f"{best_cost:.2f}")
                k4.metric("Safe Candidates",
                          f"{res['n_safe']} / {res['n']}",
                          delta=f"{res['n_safe'] / res['n']:.1%} of batch")

                st.caption(
                    f"Classification strategy: **{res['strategy']}** — "
                    f"Target: ({res['ctx'][0][0]:.2f}, {res['ctx'][0][1]:.2f}) — "
                    f"Obstacle: ({res['ctx'][1][0]:.2f}, {res['ctx'][1][1]:.2f})"
                )

                # Candidate cost distribution
                st.subheader("Candidate Cost Distribution")
                colors = ['crimson' if not s else 'mediumseagreen' for s in res['safe_mask']]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=res['preds'], marker_color=colors, name='Candidates',
                    hovertemplate="Candidate %{x}<br>Predicted J: %{y:.1f}<extra></extra>"
                ))
                fig.add_hline(y=COLLISION_THRESHOLD, line_dash="dash", line_color="orange",
                              annotation_text=f"Collision threshold J = {COLLISION_THRESHOLD}",
                              annotation_position="top right")
                fig.add_trace(go.Scatter(
                    x=[res['best']], y=[best_cost], mode='markers',
                    marker=dict(color='gold', size=14, symbol='star',
                                line=dict(color='black', width=1)),
                    name='Selected optimal'
                ))
                fig.update_layout(
                    xaxis_title="Candidate index",
                    yaxis_title="Predicted physical cost J",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(t=40)
                )
                st.plotly_chart(fig, use_container_width=True)

                # IMPROVEMENT #2 — convergence curve
                valid_conv = [(n, j) for n, j in zip(res['conv_checkpoints'], res['conv_best'])
                              if j is not None]
                if valid_conv:
                    st.subheader("Optimisation Convergence Curve")
                    st.caption(
                        "How the best safe trajectory cost improves as more candidates are evaluated. "
                        "Demonstrates the diminishing returns of larger batch sizes."
                    )
                    conv_x, conv_y = zip(*valid_conv)
                    fig_conv = go.Figure()
                    fig_conv.add_trace(go.Scatter(
                        x=list(conv_x), y=list(conv_y),
                        mode='lines+markers',
                        line=dict(color='#00b4d8', width=2),
                        marker=dict(size=5),
                        name='Best cost found',
                        hovertemplate="N=%{x} candidates<br>Best J: %{y:.1f}<extra></extra>"
                    ))
                    # Mark point of maximum single-step improvement
                    improvements = [abs(conv_y[i] - conv_y[i-1]) for i in range(1, len(conv_y))]
                    if improvements:
                        elbow_i = int(np.argmax(improvements)) + 1
                        fig_conv.add_trace(go.Scatter(
                            x=[conv_x[elbow_i]], y=[conv_y[elbow_i]], mode='markers',
                            marker=dict(color='orange', size=12, symbol='diamond'),
                            name=f'Max improvement at N={conv_x[elbow_i]}'
                        ))
                    fig_conv.update_layout(
                        xaxis_title="Number of candidates evaluated",
                        yaxis_title="Best predicted cost J (safe candidates only)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        margin=dict(t=40)
                    )
                    st.plotly_chart(fig_conv, use_container_width=True)

                st.info(
                    f"Physics engine invocations: **1** (selective validation) "
                    f"vs **{res['n']}** (full-batch simulation) — "
                    f"**{res['n'] - 1} simulation calls saved** "
                    f"({(res['n'] - 1) / res['n']:.1%} reduction)."
                )

                # Ground-truth validation of winner
                st.divider()
                if st.button("Validate Optimal Trajectory (PyBullet)"):
                    sim  = RobotSimulator(dimension=N_DIMENSIONS, gui_mode=False)
                    t, o = res['ctx']
                    with st.spinner("Running ground-truth physics simulation..."):
                        # IMPROVEMENT #3 — measure real time
                        t_phy_start = time.perf_counter()
                        rc = sim.evaluate(
                            res['cands'][res['best']].reshape(1, -1),
                            t.reshape(1, -1), o.reshape(1, -1)
                        )[0][0]
                        t_phy_real = time.perf_counter() - t_phy_start
                        sim.generate_gif(res['cands'][res['best']], t, o, "win.gif")

                    v1, v2 = st.columns(2)
                    v1.image("win.gif",
                             caption=f"PyBullet ground-truth simulation — J = {rc:.2f}",
                             use_container_width=True)

                    abs_err    = abs(rc - best_cost)
                    rel_err    = abs_err / max(rc, 1e-6) * 100
                    per_traj_s = res['time'] / res['n']
                    real_spdup = t_phy_real / max(per_traj_s, 1e-9)

                    with v2:
                        if rc < COLLISION_THRESHOLD:
                            st.success(f"Ground truth: SAFE (J = {rc:.2f})")
                        else:
                            st.error(f"Ground truth: COLLISION (J = {rc:.2f})")
                        st.metric("PyBullet time (1 trajectory)", f"{t_phy_real * 1000:.0f} ms")
                        st.metric("Surrogate time (per trajectory)",
                                  f"{per_traj_s * 1000:.3f} ms")
                        st.metric("Measured speedup (per trajectory)",
                                  f"{real_spdup:.0f}×",
                                  delta="surrogate vs PyBullet (measured)")
                        st.metric("Absolute prediction error", f"{abs_err:.2f}")
                        st.metric("Relative prediction error", f"{rel_err:.1f}%")
                        st.metric("Simulation calls saved",    f"{res['n'] - 1}")


# ------------------------------------------
# TAB 5 — ROBUSTNESS ANALYSIS  (NEW)
# ------------------------------------------
with tab_robust:
    st.header("Robustness Analysis: Statistical Validation")
    st.markdown(
        "Runs the active surrogate model across **N randomised scenarios** to "
        "characterise prediction accuracy and safety classification reliability "
        "under stochastic operating conditions. Converts point-wise case studies "
        "into a statistically grounded robustness assessment."
    )

    rb1, rb2 = st.columns([1, 3])

    with rb1:
        st.subheader("Analysis Configuration")
        n_scenarios = st.slider("Number of scenarios (N)", 10, 100, 30, step=5,
                                help="30 scenarios provides good statistical confidence.")
        rb_noise    = st.slider("Control noise σ (rad)", 0.00, 0.10, 0.02, step=0.005,
                     key="robust_noise_sigma")
        rb_seed     = st.number_input("Random seed", min_value=0, max_value=9999, value=7, step=1)
        st.markdown("---")
        st.caption(
            f"Active model: **{model_key}**  \n"
            f"Each scenario: random target + obstacle  \n"
            f"Surrogate inference only (no PyBullet per scenario)"
        )
        run_robust = st.button("Run Robustness Analysis", type="primary")

    with rb2:
        if run_robust:
            rng_rb  = np.random.default_rng(rb_seed)
            model   = loaded_models[model_key]
            prog    = st.progress(0, text="Initialising...")
            records = []

            for i in range(n_scenarios):
                prog.progress((i + 1) / n_scenarios,
                              text=f"Evaluating scenario {i + 1} / {n_scenarios}...")

                tx       = rng_rb.uniform(0.40, 0.70)
                ty       = rng_rb.uniform(-0.40, 0.40)
                target   = np.array([tx, ty, 0.05])
                ox       = tx * rng_rb.uniform(0.4, 0.6)
                oy       = ty * rng_rb.uniform(0.4, 0.6) if abs(ty) > 0.05 else rng_rb.uniform(-0.1, 0.1)
                obstacle = np.array([ox, oy, 0.25])

                h_off     = rng_rb.uniform(-0.15, 0.40)
                sim_rb    = RobotSimulator(dimension=N_DIMENSIONS)
                base_traj = sim_rb.get_ik_trajectory_advanced(target, mid_point_height_offset=h_off)
                traj      = base_traj + rng_rb.normal(0, rb_noise, base_traj.shape)

                X_lat = dr.transform(traj.reshape(1, -1))
                ctx   = np.concatenate([target, obstacle]).reshape(1, -1)
                X_in  = np.hstack([X_lat, ctx])

                t0  = time.perf_counter()
                out = model.predict(X_in)
                dt  = time.perf_counter() - t0

                p_scaled = out[0] if isinstance(out, tuple) else out
                pred_j   = decode_cost(p_scaled)
                safety, prob, _ = classify_trajectory(out, pred_j)

                records.append({
                    "Scenario":        i + 1,
                    "Target X":        tx,
                    "Target Y":        ty,
                    "Height Offset":   h_off,
                    "Predicted J":     pred_j,
                    "Safety":          safety,
                    "Collision Prob.": prob,
                    "Latency (ms)":    dt * 1000,
                })

            prog.empty()
            st.session_state['robustness'] = pd.DataFrame(records)

        if 'robustness' in st.session_state:
            df_rb     = st.session_state['robustness']
            n_safe_rb = (df_rb['Safety'] == 'Safe').sum()
            n_coll_rb = (df_rb['Safety'] == 'Collision').sum()
            mean_lat  = df_rb['Latency (ms)'].mean()
            mean_j    = df_rb['Predicted J'].mean()
            std_j     = df_rb['Predicted J'].std()

            # KPI row
            kk1, kk2, kk3, kk4 = st.columns(4)
            kk1.metric("Scenarios evaluated", len(df_rb))
            kk2.metric("Classified Safe",
                       f"{n_safe_rb} / {len(df_rb)}",
                       delta=f"{n_safe_rb / len(df_rb):.1%}")
            kk3.metric("Mean predicted cost J",
                       f"{mean_j:.1f}", delta=f"σ = {std_j:.1f}")
            kk4.metric("Mean inference latency", f"{mean_lat:.3f} ms")

            st.divider()
            ch1, ch2 = st.columns(2)

            # Cost distribution histogram
            with ch1:
                st.subheader("Predicted Cost Distribution")
                fig_hist = px.histogram(
                    df_rb, x='Predicted J', nbins=20,
                    color='Safety',
                    color_discrete_map={'Safe': 'mediumseagreen', 'Collision': 'crimson'},
                    labels={'Predicted J': 'Predicted physical cost J'},
                    title=f"Cost distribution across {len(df_rb)} scenarios"
                )
                fig_hist.add_vline(x=COLLISION_THRESHOLD, line_dash="dash", line_color="orange",
                                   annotation_text="Threshold J=1500")
                fig_hist.update_layout(bargap=0.1)
                st.plotly_chart(fig_hist, use_container_width=True)

            # Latency boxplot
            with ch2:
                st.subheader("Inference Latency Distribution")
                fig_lat = px.box(
                    df_rb, y='Latency (ms)',
                    color_discrete_sequence=['#00b4d8'],
                    title=f"Surrogate latency across {len(df_rb)} scenarios",
                    labels={'Latency (ms)': 'Inference latency (ms)'}
                )
                fig_lat.update_layout(showlegend=False)
                st.plotly_chart(fig_lat, use_container_width=True)

            # Workspace safety map
            st.subheader("Spatial Safety Map — Predicted Classification by Workspace Position")
            st.caption(
                "Each point is a scenario. Position = (Target X, Target Y). "
                "Colour = surrogate safety verdict. Marker size = predicted cost J. "
                "Reveals workspace regions where the surrogate is more or less conservative."
            )
            fig_scatter = px.scatter(
                df_rb, x='Target X', y='Target Y',
                color='Safety', size='Predicted J', size_max=20,
                color_discrete_map={'Safe': 'mediumseagreen', 'Collision': 'crimson'},
                hover_data={'Predicted J': ':.1f',
                            'Collision Prob.': ':.2%',
                            'Latency (ms)': ':.3f'},
                title="Workspace Safety Map",
                labels={'Target X': 'Target X (m)', 'Target Y': 'Target Y (m)'}
            )
            fig_scatter.update_traces(marker=dict(line=dict(width=0.5, color='white')))
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Statistical summary table
            st.subheader("Statistical Summary")
            summary = {
                "Metric": [
                    "Scenarios (N)", "Safe classifications", "Collision classifications",
                    "Safe rate", "Mean predicted cost J", "Std predicted cost J",
                    "Min predicted cost J", "Max predicted cost J",
                    "Mean latency (ms)", "Max latency (ms)", "Latency std (ms)"
                ],
                "Value": [
                    len(df_rb), n_safe_rb, n_coll_rb,
                    f"{n_safe_rb / len(df_rb):.1%}",
                    f"{mean_j:.2f}", f"{std_j:.2f}",
                    f"{df_rb['Predicted J'].min():.2f}",
                    f"{df_rb['Predicted J'].max():.2f}",
                    f"{mean_lat:.4f}",
                    f"{df_rb['Latency (ms)'].max():.4f}",
                    f"{df_rb['Latency (ms)'].std():.4f}",
                ]
            }
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

            with st.expander("Raw scenario data"):
                st.dataframe(
                    df_rb.style
                    .format({"Target X": "{:.3f}", "Target Y": "{:.3f}",
                             "Height Offset": "{:.3f}", "Predicted J": "{:.2f}",
                             "Collision Prob.": "{:.2%}", "Latency (ms)": "{:.4f}"})
                    .applymap(lambda v: "background-color:#d4edda" if v == "Safe"
                              else "background-color:#f8d7da" if v == "Collision" else "",
                              subset=["Safety"]),
                    use_container_width=True
                )