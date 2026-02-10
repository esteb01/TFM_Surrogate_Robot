import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
import time
from PIL import Image

# --- PROJECT IMPORTS ---
from src.simulation import RobotSimulator
from src.preprocessing import DimensionalityReducer
from src.surrogate_models import KrigingSurrogate, NeuralSurrogate, RBFSurrogate, SVRSurrogate, PhysicsGuidedSurrogate

# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Digital Twin: Robotic Surrogate Optimization",
    layout="wide",
    page_icon="🦾",
    initial_sidebar_state="expanded"
)

# Constants
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
N_DIMENSIONS = 350
LATENT_DIM = 16
CONTEXT_DIM = 6
COLLISION_THRESHOLD = 1500

# CSS Styling
st.markdown("""
    <style>
    .metric-card { background-color: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦾 Digital Twin: Physics-Informed Robotic Optimization")

# ==========================================
# 2. SYSTEM LOADING
# ==========================================
st.sidebar.header("Configuration")

# Model Selector
model_key = st.sidebar.selectbox(
    "Active Inference Engine",
    ("PINN (Physics-Guided)", "Neural Network (Multi-Task)", "Kriging (Standard)", "SVR (Sklearn)", "RBF (SMT)"),
    help="Select the surrogate model architecture for real-time predictions."
)


@st.cache_resource
def load_artifacts():
    try:
        dr = DimensionalityReducer(method='autoencoder', n_components=LATENT_DIM)
        dr.load(os.path.join(MODELS_DIR, 'autoencoder.pkl'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'target_scaler.pkl'))

        models = {}
        # Configuration for loading all models
        configs = [
            ("Kriging (Standard)", "kriging.pkl", KrigingSurrogate),
            ("Neural Network (Multi-Task)", "neural_net.pkl", lambda: NeuralSurrogate(LATENT_DIM + CONTEXT_DIM)),
            ("PINN (Physics-Guided)", "pinn.pkl", lambda: PhysicsGuidedSurrogate(LATENT_DIM + CONTEXT_DIM)),
            ("RBF (SMT)", "rbf.pkl", RBFSurrogate),
            ("SVR (Sklearn)", "svr.pkl", SVRSurrogate)
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
    st.error("System Error: Artifacts not found. Please run 'main.py' to initialize the system.")
    st.stop()

# ==========================================
# 3. INTERFACE TABS
# ==========================================
tab_reg, tab_cls, tab_unit, tab_opt = st.tabs([
    "📉 Regression Analysis",
    "🛡️ Safety Classification",
    "🧪 Unit Testing",
    "🏭 Industrial Optimization"
])

# Load Results CSV
res_csv = os.path.join(RESULTS_DIR, 'model_comparison.csv')
if os.path.exists(res_csv):
    df_res = pd.read_csv(res_csv)
else:
    df_res = None

# --- TAB 1: REGRESSION METRICS ---
with tab_reg:
    st.header("Performance Estimation (Regression)")
    st.markdown(
        "Evaluation of model capacity to predict continuous physical cost ($J$). Focus on energy efficiency and smoothness.")

    if df_res is not None:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### 📊 Metrics Table")
            reg_cols = ['Model', 'r2', 'rmse', 'mae']
            st.dataframe(df_res[reg_cols].style.highlight_max(axis=0, subset=['r2'], color='#d1e7dd'),
                         use_container_width=True)

            best_reg = df_res.loc[df_res['r2'].idxmax()]
            st.info(f"🏆 Best Regressor: **{best_reg['Model']}** ($R^2$: {best_reg['r2']:.4f})")

        with c2:
            st.markdown("### 📈 Precision Comparison")
            fig = px.bar(df_res, x='Model', y='r2', color='Model', text_auto='.3f', title="R2 Score (Higher is Better)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### 🔍 Residual Analysis")
    img_reg = os.path.join(RESULTS_DIR, 'regression_plots.png')
    if os.path.exists(img_reg):
        st.image(img_reg, caption="Predicted vs Actual Cost (Log Scale)", use_container_width=True)

# --- TAB 2: SAFETY METRICS ---
with tab_cls:
    st.header("Safety Assessment (Classification)")
    st.markdown(
        "Evaluation of collision detection capabilities. **Recall** is the critical metric here (avoiding false negatives).")

    if df_res is not None:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### 🛡️ Classification Metrics")
            cls_cols = ['Model', 'accuracy', 'recall', 'precision', 'f1']
            st.dataframe(df_res[cls_cols].style.highlight_max(axis=0, subset=['recall'], color='#ffcccc'),
                         use_container_width=True)

            best_safe = df_res.loc[df_res['recall'].idxmax()]
            st.error(f"🛡️ Safest Model: **{best_safe['Model']}** (Recall: {best_safe['recall']:.1%})")

        with c2:
            st.markdown("### 🚨 Recall Sensitivity")
            fig = px.bar(df_res, x='Model', y='recall', color='Model', text_auto='.3f',
                         title="Recall Score (Ability to detect crashes)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### 🧩 Confusion Matrices")
    img_cm = os.path.join(RESULTS_DIR, 'confusion_matrices.png')
    if os.path.exists(img_cm):
        st.image(img_cm, caption="True Label vs Predicted Label", use_container_width=True)

# --- TAB 3: UNIT TEST ---
with tab_unit:
    st.header("Single Scenario Validation")

    c_ctrl, c_view = st.columns([1, 2])
    with c_ctrl:
        st.markdown("### Simulation Controls")
        if st.button("🎲 Generate Random Scenario"):
            sim = RobotSimulator(dimension=N_DIMENSIONS)

            tx = np.random.uniform(0.4, 0.7);
            ty = np.random.uniform(-0.4, 0.4)
            target = np.array([tx, ty, 0.05])

            ox = tx * np.random.uniform(0.4, 0.6);
            oy = ty * np.random.uniform(0.4, 0.6)
            obstacle = np.array([ox, oy, 0.25])

            h_offset = np.random.uniform(-0.15, 0.4)
            base_traj = sim.get_ik_trajectory_advanced(target, mid_point_height_offset=h_offset)
            traj = base_traj + np.random.normal(0, 0.02, base_traj.shape)

            st.session_state['u_data'] = {'traj': traj, 'target': target, 'obs': obstacle, 'real': None}

            # Predict
            X_lat = dr.transform(traj.reshape(1, -1))
            ctx = np.concatenate([target, obstacle]).reshape(1, -1)
            X_in = np.hstack([X_lat, ctx])

            preds = []
            for name, model in loaded_models.items():
                t0 = time.time()
                res = model.predict(X_in)
                dt = time.time() - t0

                # Multi-Task Logic
                if isinstance(res, tuple):
                    p_scaled, p_prob = res
                    prob = p_prob[0][0]
                    # Logic: If Prob > 0.5 -> Collision
                    p_safe = "⚠️ Collision" if prob > 0.5 else "✅ Safe"
                    # Add prob to display string for debugging
                    p_safe += f" ({prob:.0%})"
                else:
                    p_scaled = res
                    # Heuristic for Standard Models
                    temp_cost = np.expm1(target_scaler.inverse_transform(p_scaled.reshape(-1, 1))[0][0])
                    p_safe = "⚠️ Collision" if temp_cost > COLLISION_THRESHOLD else "✅ Safe"
                    p_safe += " (Heuristic)"

                p_real = np.expm1(target_scaler.inverse_transform(p_scaled.reshape(-1, 1))[0][0])

                preds.append({"Model": name, "Pred Cost": p_real, "Safety Status": p_safe, "Latency (s)": dt})

            st.session_state['u_preds'] = pd.DataFrame(preds)

    with c_view:
        if 'u_data' in st.session_state:
            u = st.session_state['u_data']
            st.info(f"Target at ({u['target'][0]:.2f}, {u['target'][1]:.2f})")

            st.dataframe(st.session_state['u_preds'].style.format({"Pred Cost": "{:.2f}", "Latency (s)": "{:.6f}"}),
                         use_container_width=True)

            if st.button("⚖️ Validate (PyBullet)"):
                sim = RobotSimulator(dimension=N_DIMENSIONS, gui_mode=False)
                with st.spinner("Simulating Physics..."):
                    rc = sim.evaluate(u['traj'].reshape(1, -1), u['target'].reshape(1, -1), u['obs'].reshape(1, -1))[0][
                        0]
                    sim.generate_gif(u['traj'], u['target'], u['obs'], "unit.gif")
                st.session_state['u_data']['real'] = rc

            if st.session_state['u_data']['real']:
                rc = st.session_state['u_data']['real']
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.image("unit.gif", caption=f"Real Cost: {rc:.2f}", use_container_width=True)
                with c2:
                    if rc > COLLISION_THRESHOLD:
                        st.error(f"💥 COLLISION DETECTED")
                    else:
                        st.success(f"✅ SAFE TRAJECTORY")

                    # CORREGIDO: Mostrar error numérico siempre
                    model_pred = st.session_state['u_preds'].loc[
                        st.session_state['u_preds']['Model'] == model_key, 'Pred Cost'].values[0]
                    err = abs(rc - model_pred)
                    st.metric("Prediction Error", f"{err:.2f}")

# --- TAB 4: INDUSTRIAL OPTIMIZATION ---
with tab_opt:
    st.header("Real-Time Path Optimization")
    st.markdown(
        "Simulates an industrial scenario where the robot must select the optimal trajectory from **N** candidates in milliseconds.")

    col1, col2 = st.columns([1, 3])
    with col1:
        n_cand = st.slider("Candidates Batch Size", 50, 2000, 500)

        if st.button("🚀 Run Optimizer", type="primary"):
            sim = RobotSimulator(dimension=N_DIMENSIONS)
            target = np.array([0.65, 0.0, 0.05]);
            obstacle = np.array([0.35, 0.0, 0.25])

            # Generate Candidates
            cands = []
            for _ in range(n_cand):
                r = np.random.rand()
                h = np.random.uniform(-0.15, 0.05) if r < 0.4 else (
                    np.random.uniform(0.2, 0.6) if r < 0.8 else np.random.uniform(-0.1, 0.6))
                traj = sim.get_ik_trajectory_advanced(target, mid_point_height_offset=h) + np.random.normal(0, 0.015,
                                                                                                            (350,))
                cands.append(traj)
            cands = np.array(cands)

            # Inference
            model = loaded_models[model_key]
            t0 = time.time()

            X_lat = dr.transform(cands)
            ctx = np.tile(np.concatenate([target, obstacle]), (n_cand, 1))
            res = model.predict(np.hstack([X_lat, ctx]))

            # Selection Logic
            if isinstance(res, tuple):
                p_s, p_prob = res
                preds = np.expm1(target_scaler.inverse_transform(p_s).flatten())
                safe_mask = p_prob.flatten() < 0.5
                valid_costs = preds.copy()
                valid_costs[~safe_mask] = np.inf
                best_idx = np.argmin(valid_costs)
            else:
                p_s = res
                preds = np.expm1(target_scaler.inverse_transform(p_s).flatten())
                # Heuristic Filter
                valid_costs = preds.copy()
                valid_costs[preds > COLLISION_THRESHOLD] = np.inf
                best_idx = np.argmin(valid_costs)

            t_ai = time.time() - t0

            st.session_state['opt'] = {
                'cands': cands, 'preds': preds, 'best': best_idx,
                'ctx': (target, obstacle), 'time': t_ai, 'n': n_cand
            }

    with col2:
        if 'opt' in st.session_state:
            res = st.session_state['opt']
            best_cost = res['preds'][res['best']]

            k1, k2, k3 = st.columns(3)
            k1.metric("Inference Time", f"{res['time'] * 1000:.1f} ms")
            k2.metric("Optimal Cost", f"{best_cost:.2f}")

            t_phy_est = res['n'] * 0.045
            speedup = t_phy_est / max(res['time'], 1e-9)
            k3.metric("Speedup", f"{speedup:.0f}x", delta="vs Physics Engine")

            # Formula reference
            st.latex(r"Speedup = \frac{T_{Physics} \times N}{T_{AI}}")

            fig = go.Figure()
            colors = ['crimson' if c > COLLISION_THRESHOLD else 'mediumseagreen' for c in res['preds']]
            fig.add_trace(go.Bar(y=res['preds'], marker_color=colors, name='Candidates'))
            fig.add_trace(go.Scatter(x=[res['best']], y=[best_cost], mode='markers',
                                     marker=dict(color='yellow', size=15, symbol='star'), name='Selected'))
            st.plotly_chart(fig, use_container_width=True)

            if st.button("🎥 Validate Winner"):
                sim = RobotSimulator(dimension=N_DIMENSIONS, gui_mode=False)
                t, o = res['ctx']
                with st.spinner("Simulating..."):
                    rc = sim.evaluate(res['cands'][res['best']].reshape(1, -1), t.reshape(1, -1), o.reshape(1, -1))[0][
                        0]
                    sim.generate_gif(res['cands'][res['best']], t, o, "win.gif")

                c_vid, c_dat = st.columns([1, 1])
                c_vid.image("win.gif", caption=f"Real Cost: {rc:.2f}", use_container_width=True)

                err = abs(rc - best_cost)
                c_dat.metric("Prediction Error", f"{err:.2f}")
                if rc < COLLISION_THRESHOLD:
                    c_dat.success("✅ Success: Obstacle Avoided")
                else:
                    c_dat.error("❌ Failure: Collision Occurred")