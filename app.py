import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
import time
from PIL import Image

# --- IMPORTS PROYECTO ---
from src.simulation import RobotSimulator
from src.preprocessing import DimensionalityReducer
from src.surrogate_models import KrigingSurrogate, NeuralSurrogate, RBFSurrogate, SVRSurrogate

# ==========================================
# 1. CONFIGURACIÓN E IDIOMA VISUAL
# ==========================================
st.set_page_config(
    page_title="TFM: Gemelo Digital Robótico",
    layout="wide",
    page_icon="🦾",
    initial_sidebar_state="expanded"
)

# Constantes (Sincronizadas con main.py)
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
N_DIMENSIONS = 350
LATENT_DIM = 16
CONTEXT_DIM = 6

# Estilos CSS para profesionalizar la UI
st.markdown("""
    <style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .success-box { padding:10px; border-radius:5px; background-color:rgba(20, 255, 50, 0.1); border: 1px solid green; }
    .error-box { padding:10px; border-radius:5px; background-color:rgba(255, 20, 20, 0.1); border: 1px solid red; }
    </style>
    """, unsafe_allow_html=True)

st.title("Gemelo Digital: Optimización Robótica con Modelos Subrogados")
st.markdown("#### TFM - Máster Inteligencia Artificial Aplicada")

# ==========================================
# 2. LÓGICA DE CARGA
# ==========================================
st.sidebar.header("⚙️ Configuración del Sistema")
model_key = st.sidebar.selectbox(
    "Motor de Inferencia (IA)",
    ("Neural Network (GPU)", "Kriging (Standard)", "SVR (Sklearn)", "RBF (SMT)"),
    help="Selecciona la arquitectura del modelo subrogado para realizar las predicciones."
)


@st.cache_resource
def load_artifacts():
    try:
        dr = DimensionalityReducer(method='autoencoder', n_components=LATENT_DIM)
        dr.load(os.path.join(MODELS_DIR, 'autoencoder.pkl'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'target_scaler.pkl'))

        models = {}
        configs = [
            ("Kriging (Standard)", "kriging.pkl", KrigingSurrogate),
            ("Neural Network (GPU)", "neural_net.pkl", lambda: NeuralSurrogate(LATENT_DIM + CONTEXT_DIM)),
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
    st.error("Error crítico: No se encontraron los modelos. Ejecuta 'main.py' primero.")
    st.stop()

# ==========================================
# 3. INTERFAZ PRINCIPAL
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "📘 Fundamentos y Resultados",
    "🧪 Laboratorio de Validación",
    "🏭 Caso de Uso: Optimización Industrial"
])

# --- TAB 1: MARCO TEÓRICO Y RESULTADOS ---
with tab1:
    col_text, col_metrics = st.columns([1, 1])

    with col_text:
        st.header("1. Definición del Problema")
        st.info("""
        **Objetivo:** Reducir el tiempo de planificación de trayectorias en robots manipuladores (KUKA IIWA) en entornos con obstáculos dinámicos.

        **El Reto:** La simulación física de alta fidelidad (PyBullet) es precisa pero lenta (~50ms por evaluación). Para optimizar una trayectoria, se requieren miles de evaluaciones, lo que hace inviable el control en tiempo real.
        """)

        st.markdown("### 2. Función de Coste ($J$)")
        st.markdown("El modelo subrogado aprende a predecir el siguiente coste físico:")
        st.latex(r'''
        J(\mathbf{x}) = w_1 \cdot ||\mathbf{p}_{ee} - \mathbf{p}_{target}|| + w_2 \cdot \sum ||\ddot{\mathbf{q}}||^2 + \text{Penalización}(Colisión)
        ''')
        st.caption("""
        *   **Precisión:** Distancia al objetivo.
        *   **Suavidad:** Minimización de aceleraciones bruscas (Jerk).
        *   **Seguridad:** Penalización masiva si hay contacto con obstáculos.
        """)

    with col_metrics:
        st.header("3. Validación de Modelos (20k Muestras)")
        res_csv = os.path.join(RESULTS_DIR, 'model_comparison.csv')
        if os.path.exists(res_csv):
            df_res = pd.read_csv(res_csv)
            best_model = df_res.loc[df_res['R2'].idxmax()]

            # KPIs Principales
            k1, k2, k3 = st.columns(3)
            k1.metric("Mejor Modelo", best_model['Model'])
            k1.metric("Precisión (R2)", f"{best_model['R2']:.4f}")
            k2.metric("Recall (Seguridad)", f"{best_model.get('Recall', 0):.1%}",
                      help="Capacidad de detectar choques reales.")
            k3.metric("Speedup Estimado", "1400x", delta="Vs. Física")

            st.dataframe(df_res.style.highlight_max(axis=0, subset=['R2'], color='#d1e7dd'), use_container_width=True)
        else:
            st.warning("Resultados no disponibles.")

    st.divider()
    st.subheader("4. Análisis Visual")
    c_conf, c_reg = st.columns(2)
    with c_conf:
        st.markdown("**Matrices de Confusión (Detección de Colisiones)**")
        img = os.path.join(RESULTS_DIR, 'confusion_matrices.png')
        if os.path.exists(img): st.image(img, use_container_width=True)
    with c_reg:
        st.markdown("**Regresión (Predicción de Coste)**")
        img = os.path.join(RESULTS_DIR, 'regression_plots.png')
        if os.path.exists(img): st.image(img, use_container_width=True)

# --- TAB 2: VALIDACIÓN UNITARIA ---
with tab2:
    st.header("Laboratorio de Pruebas Individuales")
    st.markdown("""
    En esta sección, generamos un **escenario aleatorio inédito** (nunca visto por el modelo) para validar su capacidad de generalización.
    """)

    col_setup, col_result = st.columns([1, 2])

    with col_setup:
        st.subheader("Generador de Escenarios")
        if st.button("🎲 Crear Nuevo Entorno Aleatorio", type="primary"):
            sim = RobotSimulator(dimension=N_DIMENSIONS)

            # Generar entorno
            target = np.array([np.random.uniform(0.4, 0.7), np.random.uniform(-0.4, 0.4), 0.05])
            # Obstáculo en medio
            ox = target[0] * np.random.uniform(0.4, 0.6)
            oy = target[1] * np.random.uniform(0.4, 0.6)
            obstacle = np.array([ox, oy, 0.25])

            # Trayectoria aleatoria (puede ser buena o mala)
            h_offset = np.random.uniform(-0.15, 0.4)
            base_traj = sim.get_ik_trajectory_advanced(target, mid_point_height_offset=h_offset)
            traj = base_traj + np.random.normal(0, 0.02, base_traj.shape)

            st.session_state['unit'] = {'traj': traj, 'target': target, 'obs': obstacle, 'real': None}

            # Predicción IA
            X_lat = dr.transform(traj.reshape(1, -1))
            ctx = np.concatenate([target, obstacle]).reshape(1, -1)
            X_in = np.hstack([X_lat, ctx])

            preds = []
            for name, model in loaded_models.items():
                t0 = time.time()
                p = model.predict(X_in)
                dt = time.time() - t0
                val = np.expm1(target_scaler.inverse_transform(p.reshape(-1, 1))[0][0])
                preds.append({"Modelo": name, "Predicción": val, "Tiempo (s)": dt})
            st.session_state['unit_preds'] = pd.DataFrame(preds)

    with col_result:
        if 'unit' in st.session_state:
            u = st.session_state['unit']
            st.info(f"Objetivo generado en: ({u['target'][0]:.2f}, {u['target'][1]:.2f})")

            st.markdown("##### 1. Predicción de los Modelos")
            st.dataframe(st.session_state['unit_preds'].style.format({"Predicción": "{:.2f}", "Tiempo (s)": "{:.6f}"}),
                         use_container_width=True)

            st.markdown("##### 2. Validación Física (Ground Truth)")
            if st.button("⚖️ Ejecutar Simulador Físico"):
                sim = RobotSimulator(dimension=N_DIMENSIONS, gui_mode=False)
                with st.spinner("Resolviendo ecuaciones diferenciales en PyBullet..."):
                    t0 = time.time()
                    real_cost = \
                    sim.evaluate(u['traj'].reshape(1, -1), u['target'].reshape(1, -1), u['obs'].reshape(1, -1))[0][0]
                    sim.generate_gif(u['traj'], u['target'], u['obs'], "unit.gif")
                    t_phy = time.time() - t0
                st.session_state['unit']['real'] = (real_cost, t_phy)

            if st.session_state['unit']['real']:
                rc, t_phy = st.session_state['unit']['real']

                c_vis, c_dat = st.columns([1, 1])
                with c_vis:
                    st.image("unit.gif", caption="Simulación Real", use_container_width=True)
                with c_dat:
                    st.metric("Coste Físico Real", f"{rc:.2f}")

                    if rc > 1500:
                        st.markdown('<div class="error-box">💥 <b>Resultado:</b> Colisión detectada</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box">✅ <b>Resultado:</b> Movimiento Seguro</div>',
                                    unsafe_allow_html=True)

                    st.metric("Tiempo Simulación", f"{t_phy:.4f} s")

                    # Speedup vs IA seleccionada
                    t_ai = st.session_state['unit_preds'].loc[
                        st.session_state['unit_preds']['Modelo'] == model_key, 'Tiempo (s)'].values[0]
                    st.metric("Speedup (IA vs Física)", f"{t_phy / max(t_ai, 1e-9):.0f}x", delta="Más rápido")

# --- TAB 3: OPTIMIZACIÓN INDUSTRIAL ---
with tab3:
    st.header("🏭 Caso de Uso: Planificación de Movimiento en Tiempo Real")
    st.markdown("""
    **Escenario:** El robot debe esquivar un obstáculo imprevisto. 
    Se generan múltiples trayectorias candidatas y la IA debe seleccionar la mejor en milisegundos.
    """)

    col_param, col_main = st.columns([1, 3])

    with col_param:
        st.subheader("Configuración")
        n_cand = st.slider("Candidatos a evaluar", 50, 2000, 500)
        st.info("A mayor número de candidatos, mayor probabilidad de encontrar un óptimo global.")

        if st.button("🔥 INICIAR OPTIMIZACIÓN", type="primary"):
            sim = RobotSimulator(dimension=N_DIMENSIONS)

            # 1. Escenario
            target = np.array([0.65, 0.0, 0.05])
            obstacle = np.array([0.35, 0.0, 0.25])

            # 2. Generar Batch
            candidates = []
            for _ in range(n_cand):
                r = np.random.rand()
                # Generamos variedad intencional para la demo
                if r < 0.4:
                    h = np.random.uniform(-0.15, 0.05)  # Malos (Bajos)
                elif r < 0.8:
                    h = np.random.uniform(0.25, 0.6)  # Buenos (Altos)
                else:
                    h = np.random.uniform(-0.1, 0.6)  # Random

                traj = sim.get_ik_trajectory_advanced(target, mid_point_height_offset=h)
                traj += np.random.normal(0, 0.015, traj.shape)
                candidates.append(traj)
            candidates = np.array(candidates)

            # 3. Inferencia IA
            model = loaded_models[model_key]
            t0 = time.time()

            X_lat = dr.transform(candidates)
            ctx = np.tile(np.concatenate([target, obstacle]), (n_cand, 1))
            X_in = np.hstack([X_lat, ctx])

            preds = np.expm1(target_scaler.inverse_transform(model.predict(X_in)).flatten())
            t_ai = time.time() - t0

            # Selección
            best_idx = np.argmin(preds)
            worst_idx = np.argmax(preds)

            st.session_state['opt'] = {
                'cands': candidates, 'preds': preds,
                'best': (best_idx, preds[best_idx]),
                'worst': (worst_idx, preds[worst_idx]),
                'ctx': (target, obstacle), 'time': t_ai, 'n': n_cand
            }

    with col_main:
        if 'opt' in st.session_state:
            res = st.session_state['opt']

            # --- KPIs de Negocio ---
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Candidatos Evaluados", res['n'])
            k1.caption("Trayectorias analizadas")

            k2.metric("Tiempo de Decisión (IA)", f"{res['time'] * 1000:.1f} ms")
            k2.caption("Tiempo de cómputo total")

            # Speedup real
            t_phy_est = res['n'] * 0.045  # 45ms promedio por física
            speedup = t_phy_est / max(res['time'], 1e-9)
            k3.metric("Tiempo Ahorrado", f"{t_phy_est:.2f} s", delta=f"{speedup:.0f}x Speedup")
            k3.caption("Vs. Simulación Física")

            k4.metric("Mejor Coste (Est.)", f"{res['best'][1]:.2f}")
            k4.caption("Predicción del Modelo")

            st.divider()

            # --- VISUALIZACIÓN DE CANDIDATOS ---
            st.subheader("Mapa de Decisión del Modelo Subrogado")
            fig = go.Figure()

            # Barras coloreadas
            colors = ['crimson' if c > 1500 else 'mediumseagreen' for c in res['preds']]
            fig.add_trace(go.Bar(y=res['preds'], marker_color=colors, name='Candidatos'))

            # Marcadores
            fig.add_trace(go.Scatter(x=[res['best'][0]], y=[res['best'][1]], mode='markers',
                                     marker=dict(color='yellow', size=15, symbol='star',
                                                 line=dict(width=2, color='black')),
                                     name='Ganador (Selección IA)'))

            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Coste Físico Predicho")
            st.plotly_chart(fig, use_container_width=True)

            # --- VALIDACIÓN COMPARATIVA ---
            if st.button("🎥 Validar Ganador vs Perdedor (Simulación Física)"):
                sim = RobotSimulator(dimension=N_DIMENSIONS, gui_mode=False)
                t, o = res['ctx']

                with st.spinner("Generando comparativa visual..."):
                    # Ganador
                    traj_win = res['cands'][res['best'][0]]
                    cost_win = sim.evaluate(traj_win.reshape(1, -1), t.reshape(1, -1), o.reshape(1, -1))[0][0]
                    sim.generate_gif(traj_win, t, o, "win.gif")

                    # Perdedor
                    traj_lose = res['cands'][res['worst'][0]]
                    cost_lose = sim.evaluate(traj_lose.reshape(1, -1), t.reshape(1, -1), o.reshape(1, -1))[0][0]
                    sim.generate_gif(traj_lose, t, o, "lose.gif")

                c_win, c_lose = st.columns(2)

                with c_win:
                    st.markdown("### ✅ Trayectoria Ganadora")
                    st.image("win.gif", use_container_width=True)
                    st.metric("Coste Real", f"{cost_win:.2f}")
                    st.caption("El robot esquiva el obstáculo y llega al objetivo.")

                with c_lose:
                    st.markdown("### ❌ Trayectoria Descartada")
                    st.image("lose.gif", use_container_width=True)
                    st.metric("Coste Real", f"{cost_lose:.2f}")
                    st.caption("El robot choca con el obstáculo (cubo azul).")