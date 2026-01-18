import streamlit as st
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

def run_monte_carlo_simulation(bankroll, cost, win_rate, avg_withdrawal, num_simulations):
    """
    Ejecuta la simulación de Montecarlo basada en la lógica de 'Riesgo de Ruina'.
    """
    ruins = 0
    total_final_bankroll = 0
    all_trajectories = []
    
    # Límite de intentos para evitar bucles infinitos en estrategias ganadoras
    MAX_ATTEMPTS_PER_SIM = 200
    
    # Progreso en la barra de Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_simulations):
        current_bank = bankroll
        attempts = 0
        history = [{'step': 0, 'balance': current_bank, 'sim_id': f'Sim {i+1}'}]
        is_ruined = False
        
        while attempts < MAX_ATTEMPTS_PER_SIM:
            # Condición de Ruina: No tienes suficiente para el siguiente intento
            if current_bank < cost:
                ruins += 1
                is_ruined = True
                break
            
            # Pagar el coste del intento (Evaluación)
            current_bank -= cost
            attempts += 1
            
            # Tirar los dados (0-100)
            roll = random.uniform(0, 100)
            
            if roll < win_rate:
                # Éxito: Sumar retiro medio
                current_bank += avg_withdrawal
            
            # Guardamos la historia para el gráfico
            history.append({'step': attempts, 'balance': current_bank, 'sim_id': f'Sim {i+1}'})
        
        all_trajectories.extend(history)
            
        total_final_bankroll += current_bank
        
        # Actualizar barra de progreso cada cierto tiempo para no ralentizar
        if i % (num_simulations // 10) == 0:
            progress = (i + 1) / num_simulations
            progress_bar.progress(progress)
            
    progress_bar.empty() # Limpiar barra al terminar
    
    return {
        'ruins': ruins,
        'risk_of_ruin': (ruins / num_simulations) * 100,
        'avg_final_bankroll': total_final_bankroll / num_simulations,
        'survived': num_simulations - ruins,
        'chart_data': pd.DataFrame(all_trajectories)
    }

def show_page():
    # Configuración de la página (solo si se ejecuta como script principal, 
    # si se importa como módulo, esto se ignora o se maneja en el main)
    try:
        st.set_page_config(page_title="Calculadora Montecarlo", page_icon="📈", layout="wide")
    except:
        pass # Por si ya se configuró en otra página

    st.title("📈 Calculadora de Riesgo de Ruina (Montecarlo)")
    
    # Importar y aplicar estilos globales
    import utils
    utils.apply_global_styles()

    st.markdown("""
    Esta herramienta simula miles de escenarios para determinar la viabilidad matemática de tu estrategia de fondeo.
    **Objetivo:** Mantener el Riesgo de Ruina por debajo del 5%.
    """)

    # --- Layout: Sidebar para Inputs ---
    with st.sidebar:
        st.header("Parámetros")
        
        bankroll = st.number_input("Bankroll Inicial (€/$)", value=3000, step=100)
        cost = st.number_input("Coste medio por intento (€/$)", value=100, step=10)
        win_rate = st.number_input("Probabilidad de Retiro (%)", value=15.0, step=0.5, format="%.1f")
        avg_withdrawal = st.number_input("Retiro Medio (€/$)", value=1000, step=50)
        
        st.markdown("---")
        
        num_simulations = st.selectbox("Nº Simulaciones", [100, 1000, 5000, 10000], index=1)
        
        # Cálculo de Esperanza Matemática (EV)
        # Fórmula desglosada: (Prob_Ganar * Beneficio_Neto) + (Prob_Perder * Pérdida_Neta)
        # Beneficio_Neto = avg_withdrawal - cost
        # Pérdida_Neta = -cost
        # Esto se simplifica matemáticamente a: (win_rate * avg_withdrawal) - cost
        p_win = win_rate / 100
        p_loss = 1 - p_win
        ev_per_attempt = (p_win * (avg_withdrawal - cost)) + (p_loss * (-cost))
        
        # El display se ha movido a la zona principal (user request)

        run_btn = st.button("Ejecutar Simulación", type="primary")

    # --- Zona Principal ---
    
    if run_btn:
        with st.spinner('Calculando escenarios...'):
            # Ejecutar lógica
            results = run_monte_carlo_simulation(bankroll, cost, win_rate, avg_withdrawal, num_simulations)
            
            # --- Métricas Principales ---
            
            # Fila 1: KPIs
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                with st.container(border=True):
                    # Riesgo de Ruina
                    risk = results['risk_of_ruin']
                    st.metric(
                        label="Riesgo de Ruina", 
                        value=f"{risk:.2f}%", 
                        delta="Meta: < 5%",
                        delta_color="off" if risk < 5 else "inverse"
                    )
                    # Estado visual asociado
                    if risk < 5:
                        st.success("✅ Seguro")
                    elif risk < 20:
                        st.warning("⚠️ Precaución")
                    else:
                        st.error("⛔ Peligro")

            with c2:
                with st.container(border=True):
                    st.metric(
                        label="Supervivencia", 
                        value=f"{results['survived']} / {num_simulations}",
                        help="Número de simulaciones que NO quebraron."
                    )
                    st.caption(f"Tasa: {(results['survived']/num_simulations)*100:.1f}%")

            with c3:
                with st.container(border=True):
                    st.metric(
                        label="Bankroll Final Medio", 
                        value=f"{results['avg_final_bankroll']:,.0f} €/$",
                        help="Promedio de capital tras 200 intentos (si sobrevive)."
                    )
            
            with c4:
                with st.container(border=True):
                    st.metric(
                        label="Esperanza Matemática (EV)",
                        value=f"{ev_per_attempt:+.2f} €/$",
                        help="Ganancia o pérdida promedio por cada intento."
                    )
                    if ev_per_attempt > 0:
                        st.success("Rentable")
                    else:
                        st.error("No Rentable")

            # --- Fórmula Explicada (Ocultable) ---
            with st.expander("ℹ️ Ver desglose matemático (EV)"):
                st.markdown(f"""
                **Fórmula de Esperanza Matemática:**
                $$ EV = (P_{{ganar}} \\times (Retiro - Coste)) + (P_{{perder}} \\times (-Coste)) $$
                
                Aplicado a tus datos:
                $$ EV = ({p_win:.2f} \\times ({avg_withdrawal} - {cost})) + ({p_loss:.2f} \\times (-{cost})) = \\mathbf{{{ev_per_attempt:+.2f}}} $$
                """)

            st.markdown("---")

            # --- Gráfico ---
            st.subheader(f"Proyección de Trayectorias ({num_simulations} simulaciones)")
            
            if not results['chart_data'].empty:
                df_chart = results['chart_data']
                
                # Crear gráfico con Plotly
                fig = px.line(
                    df_chart, 
                    x="step", 
                    y="balance", 
                    color="sim_id",
                    labels={"step": "Intentos (Evaluaciones)", "balance": "Capital (€/$)", "sim_id": "Simulación"},
                    title="Evolución del Bankroll"
                )
                
                # Añadir líneas de referencia
                fig.add_hline(y=cost, line_dash="dash", line_color="red", annotation_text="Zona de Ruina")
                fig.add_hline(y=bankroll, line_dash="dash", line_color="gray", annotation_text="Inicial")
                
                # Personalización visual
                fig.update_layout(showlegend=False, height=500)
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption(f"Nota: El gráfico muestra todas las simulaciones calculadas.")
            
    else:
        # Estado inicial (antes de pulsar el botón)
        st.info("👈 Ajusta los parámetros en la barra lateral y pulsa 'Ejecutar Simulación' para ver los resultados.")
        
        # Mostrar imagen placeholder o ejemplo visual
        st.markdown("""
        ### ¿Cómo interpretar los resultados?
        
        * **Riesgo de Ruina:** Es la probabilidad de que tu capital baje tanto que no puedas pagar la siguiente evaluación.
        * **Esperanza Matemática:** Cuánto dinero ganas (o pierdes) de media por cada intento.
        """)

if __name__ == "__main__":
    show_page()
