import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np
import io
import utils

# --- Constants & Mappings ---
ES_TO_EN_MAP = {
    'Tiempo': 'Time', 'Time': 'Time', 'Hora': 'Time', 'Fecha': 'Time', 'Fecha/Hora': 'Time',
    'Hora de apertura': 'Time', 'Ticket': 'Ticket', 'Orden': 'Ticket', 'Transacción': 'Ticket',
    'Order': 'Ticket', 'Deal': 'Ticket', 'Operación': 'Ticket', 'Tipo': 'Type',
    'Dirección': 'Direction', 'Direction': 'Direction', 'Type': 'Type', 'Volumen': 'Volume',
    'Volume': 'Volume', 'Precio': 'Price', 'Price': 'Price', 'S/L': 'S/L', 'T/P': 'T/P',
    'Beneficio': 'Profit', 'Profit': 'Profit', 'Balance': 'Balance', 'Comisión': 'Commission',
    'Comision': 'Commission', 'Commission': 'Commission', 'Swap': 'Swap', 'Comentario': 'Comment',
    'Comment': 'Comment'
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Detects if columns are in Spanish and renames them to English standard."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    current_cols = set(df.columns)
    mapped_cols = {}
    for es_col, en_col in ES_TO_EN_MAP.items():
        if es_col in current_cols:
            mapped_cols[es_col] = en_col
    if mapped_cols:
        df = df.rename(columns=mapped_cols)
    return df

def clean_money(val):
    if isinstance(val, str):
        val = val.replace(' ', '').replace(',', '.')
    return pd.to_numeric(val, errors='coerce')

def parse_mt5_html(file_buffer):
    """Parses the MT5 HTML report."""
    debug_log = []
    try:
        # Reset buffer
        if hasattr(file_buffer, 'seek'):
            file_buffer.seek(0)
            
        dfs = pd.read_html(file_buffer, header=None)
        debug_log.append(f"Tablas raw leídas: {len(dfs)}")
        
        valid_tables = []
        
        for i_df, df in enumerate(dfs):
            # Debug shape
            # search for headers
            search_df = df.astype(str)
            mask = search_df.apply(lambda col: col.str.contains('Profit|Beneficio', case=False, na=False))
            rows_with_header = mask.any(axis=1)
            
            if rows_with_header.any():
                candidate_indices = rows_with_header[rows_with_header].index.tolist()
                for header_row_idx in candidate_indices:
                    try:
                        header_row = df.iloc[header_row_idx].astype(str).tolist()
                        temp_df = df.iloc[header_row_idx+1:].copy()
                        temp_df.columns = header_row
                        temp_df = normalize_columns(temp_df)
                        
                        if 'Profit' in temp_df.columns and 'Time' in temp_df.columns:
                            valid_tables.append(temp_df)
                            break 
                    except:
                        continue

        if not valid_tables:
            return pd.DataFrame(), debug_log

        # Select largest
        target_df = max(valid_tables, key=len)
        debug_log.append(f"Tabla seleccionada: {len(target_df)} filas (Crudas)")

        target_df = target_df.copy()
        
        # Clean Profit
        target_df['Profit'] = target_df['Profit'].apply(clean_money)
        target_df = target_df.dropna(subset=['Profit'])
        target_df['Time'] = pd.to_datetime(target_df['Time'], errors='coerce')
        
        if 'Direction' in target_df.columns:
            target_df['Direction'] = target_df['Direction'].astype(str).str.lower().str.strip()
            
            # LOG DISTRIBUTION
            dist_counts = target_df['Direction'].value_counts().to_dict()
            debug_log.append(f"Distribución de Dirección (antes de filtro): {dist_counts}")
            
            # STRICT FILTERING to match notebook logic (df['Direction'] == 'out')
            # Excluding 'in/out' which are reversals
            target_df = target_df[target_df['Direction'].isin(['out', 'salida'])].copy()
            debug_log.append(f"Filtrado estricto ('out'/'salida'): {len(target_df)} operaciones.")
        else:
            debug_log.append("ADVERTENCIA: Columna 'Direction' no encontrada.")
        
        target_df = target_df.sort_values('Time').reset_index(drop=True)
        return target_df, debug_log

    except Exception as e:
        debug_log.append(f"Error parse_mt5_html: {str(e)}")
        return pd.DataFrame(), debug_log

def calculate_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates signed streaks: +1, +2 (Wins), -1, -2 (Losses)."""
    if df.empty:
        return df
    df = df.copy()
    df['Result'] = df['Profit'].apply(lambda x: 'Win' if x >= 0 else 'Loss')
    
    streak_counters = []
    current_streak = 0
    
    for res in df['Result']:
        if res == 'Win':
            if current_streak >= 0: current_streak += 1
            else: current_streak = 1
        else:
            if current_streak <= 0: current_streak -= 1
            else: current_streak = -1
        streak_counters.append(current_streak)
        
    df['Signed_Streak'] = streak_counters
    return df

def get_streak_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Compresses trade list into streak blocks."""
    if df.empty:
        return pd.DataFrame()
        
    block_streaks = []
    current_type = None
    current_count = 0
    start_idx = 0
    accumulated_profit = 0
    
    for i, row in df.iterrows():
        res = row['Result']
        profit = row['Profit']
        
        if res == current_type:
            current_count += 1
            accumulated_profit += profit
        else:
            if current_type is not None:
                block_streaks.append({
                    'Type': current_type,
                    'Count': current_count,
                    'Start_Time': df.loc[start_idx, 'Time'],
                    'End_Time': df.loc[i-1, 'Time'],
                    'Total_Profit': accumulated_profit
                })
            current_type = res
            current_count = 1
            accumulated_profit = profit
            start_idx = i
            
    # Last block
    if current_type is not None:
        block_streaks.append({
            'Type': current_type,
            'Count': current_count,
            'Start_Time': df.loc[start_idx, 'Time'],
            'End_Time': df.iloc[-1]['Time'],
            'Total_Profit': accumulated_profit
        })
        
    return pd.DataFrame(block_streaks)

def debug_mt5_html(file_buffer):
    try:
        if hasattr(file_buffer, 'seek'):
            file_buffer.seek(0)
        dfs = pd.read_html(file_buffer, header=None)
        st.write(f"Tablas encontradas: {len(dfs)}")
        for i, df in enumerate(dfs):
            st.write(f"Tabla {i}:")
            st.write(f"Dimensiones: {df.shape}")
            st.dataframe(df.head(10))
    except Exception as e:
        st.error(f"Error debug: {e}")

def show_page():
    st.set_page_config(page_title="Streak Analysis", page_icon="🔥", layout="wide")
    utils.apply_global_styles()
    
    st.title("🔥 Análisis de Rachas (Streaks)")
    st.markdown("Sube tu reporte de **MetaTrader 5 (HTML)** para analizar la distribución de tus rachas ganadoras y perdedoras.")
    
    uploaded_file = st.file_uploader("Subir reporte MT5 (.html)", type=["html", "htm"])
    
    if uploaded_file is not None:
        with st.spinner("Procesando archivo..."):
            df_trades, debug_log = parse_mt5_html(uploaded_file)
            
        if df_trades.empty:
            st.error("No se pudo encontrar una tabla de operaciones válida.")
            with st.expander("🕵️‍♂️ Debug: Ver tablas encontradas"):
                st.text("\n".join(debug_log))
                debug_mt5_html(uploaded_file)
            return
            
        # Calculate Logic
        df = calculate_streaks(df_trades)
        streak_blocks = get_streak_blocks(df)
        
        st.success(f"Reporte cargado: {len(df)} operaciones cerradas ('out'). Generadas {len(streak_blocks)} rachas.")
        
        # --- Metrics ---
        if not streak_blocks.empty:
            max_win = streak_blocks[streak_blocks['Type'] == 'Win']['Count'].max() if not streak_blocks[streak_blocks['Type'] == 'Win'].empty else 0
            max_loss = streak_blocks[streak_blocks['Type'] == 'Loss']['Count'].max() if not streak_blocks[streak_blocks['Type'] == 'Loss'].empty else 0
            avg_win = streak_blocks[streak_blocks['Type'] == 'Win']['Count'].mean() if not streak_blocks[streak_blocks['Type'] == 'Win'].empty else 0
            avg_loss = streak_blocks[streak_blocks['Type'] == 'Loss']['Count'].mean() if not streak_blocks[streak_blocks['Type'] == 'Loss'].empty else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Max Racha Ganadora", f"{max_win}")
            c2.metric("Max Racha Perdedora", f"{max_loss}", delta_color="inverse")
            c3.metric("Promedio Racha Ganadora", f"{avg_win:.2f}")
            c4.metric("Promedio Racha Perdedora", f"{avg_loss:.2f}")
        
        st.markdown("---")
        
        # --- 1. Sequential Plot ---
        st.subheader("1. Secuencia de Rachas (Cronológico)")
        
        # If we visualize TRADES sequence
        fig_seq = go.Figure()
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df['Signed_Streak']]
        
        fig_seq.add_trace(go.Bar(
            x=df.index,
            y=df['Signed_Streak'],
            marker_color=colors,
            name='Streak'
        ))
        
        fig_seq.update_layout(
            title=f"Evolución de Rachas (Por Operación)",
            xaxis_title="# Operación (Tiempo)",
            yaxis_title="Racha Consecutiva (+/-)",
            height=400
        )
        st.plotly_chart(fig_seq, use_container_width=True)
        
        # --- 2. Distributions (Pyramid) ---
        st.subheader("2. Distribución de Rachas (Win vs Loss)")
        
        if not streak_blocks.empty:
            win_counts = streak_blocks[streak_blocks['Type'] == 'Win']['Count'].value_counts().sort_index()
            loss_counts = streak_blocks[streak_blocks['Type'] == 'Loss']['Count'].value_counts().sort_index()
            
            all_lengths = sorted(list(set(win_counts.index).union(set(loss_counts.index))))
            
            win_freq = [win_counts.get(l, 0) for l in all_lengths]
            loss_freq = [-loss_counts.get(l, 0) for l in all_lengths] # Negative for pyramid
            
            fig_pyr = go.Figure()
            fig_pyr.add_trace(go.Bar(
                y=all_lengths, x=loss_freq, orientation='h', name='Losses', marker_color='#e74c3c',
                text=[str(abs(x)) for x in loss_freq], textposition='auto'
            ))
            fig_pyr.add_trace(go.Bar(
                y=all_lengths, x=win_freq, orientation='h', name='Wins', marker_color='#2ecc71',
                text=win_freq, textposition='auto'
            ))
            
            # Formatear eje X para mostrar valores positivos a ambos lados
            max_val = max(max(win_freq) if win_freq else 0, max([abs(x) for x in loss_freq]) if loss_freq else 0)
            
            fig_pyr.update_layout(
                title="Pirámide de Frecuencia de Rachas",
                barmode='overlay',
                xaxis_title="Frecuencia",
                yaxis_title="Longitud de Racha",
                xaxis=dict(
                    range=[-max_val*1.1, max_val*1.1]
                )
            )
            st.plotly_chart(fig_pyr, use_container_width=True)

        # --- 3. Normal Fit ---
        st.subheader("3. Ajuste de Distribución Normal")
        
        if not streak_blocks.empty:
            # Use Streak Blocks (Aggregated) unlike the Trade-based Signed_Streak
            # This matches the notebook logic: 
            # streak_df['Signed_Count'] = streak_df.apply(lambda x: x['Count'] if x['Type'] == 'Win' else -x['Count'], axis=1)
            streak_blocks = streak_blocks.copy()
            streak_blocks['Signed_Count'] = streak_blocks.apply(lambda x: x['Count'] if x['Type'] == 'Win' else -x['Count'], axis=1)
            data_signed = streak_blocks['Signed_Count']
            
            if len(data_signed) > 1:
                mu, std = stats.norm.fit(data_signed)
                med = np.median(data_signed)
                
                fig_dist = go.Figure()
                
                # 1. Histogram (Raw Counts)
                fig_dist.add_trace(go.Histogram(
                    x=data_signed, 
                    name='Datos Reales (Conteo)', 
                    marker_color='#90caf9',
                    opacity=0.75,
                    xbins=dict(start=int(data_signed.min())-0.5, end=int(data_signed.max())+0.5, size=1)
                ))
                
                # 2. Normal Distribution Curve (Scaled to Counts)
                xmin, xmax = data_signed.min(), data_signed.max()
                x_lin = np.linspace(xmin, xmax, 200)
                p = stats.norm.pdf(x_lin, mu, std)
                
                # Scale PDF to match counts: PDF * Total_Samples * Bin_Width
                # Bin width is 1 (integer streaks)
                p_scaled = p * len(data_signed) * 1 
                
                fig_dist.add_trace(go.Scatter(
                    x=x_lin, y=p_scaled, mode='lines', 
                    name=f'Normal (μ={mu:.2f}, σ={std:.2f})',
                    line=dict(color='#ff1744', width=3, dash='dash')
                ))
                
                # 3. Mean and Median Lines
                fig_dist.add_vline(x=mu, line_width=2, line_dash="dash", line_color="black", annotation_text=f"Media: {mu:.2f}")
                fig_dist.add_vline(x=med, line_width=2, line_dash="dot", line_color="green", annotation_text=f"Mediana: {med:.2f}")
                
                fig_dist.update_layout(
                    title="Distribución de Rachas (Frecuencia vs Normal)",
                    xaxis_title="Longitud de Racha (Negativo=Loss, Positivo=Win)",
                    yaxis_title="Frecuencia (Conteo)",
                    bargap=0.1,
                    legend=dict(x=0.01, y=0.99)
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.warning("Datos insuficientes para ajustar distribución normal.")
        
        # --- 4. Top 10 Tables ---
        st.subheader("4. Top 10 Rachas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🏆 Top 10 Rachas Ganadoras")
            if not streak_blocks.empty:
                top_wins = streak_blocks[streak_blocks['Type'] == 'Win'].nlargest(10, 'Count')
                st.dataframe(top_wins[['Count', 'Total_Profit', 'Start_Time', 'End_Time']].reset_index(drop=True), use_container_width=True)
                
        with col2:
            st.markdown("##### 💀 Top 10 Rachas Perdedoras")
            if not streak_blocks.empty:
                top_losses = streak_blocks[streak_blocks['Type'] == 'Loss'].nlargest(10, 'Count')
                st.dataframe(top_losses[['Count', 'Total_Profit', 'Start_Time', 'End_Time']].reset_index(drop=True), use_container_width=True)

if __name__ == "__main__":
    show_page()
