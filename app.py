import streamlit as st
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# --- Cargar modelo y transformadores ---
model = XGBClassifier()
model.load_model("modelo_xgboost_sorteos.json")
scaler = joblib.load("escalador_sorteos.pkl")
le = joblib.load("label_encoder_sorteos.pkl")

# --- Cargar CSV de sorteos ---
@st.cache_data
def cargar_datos():
    df = pd.read_csv("sorteos_unificado.csv")
    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'], errors='coerce')
    df['hora'] = df['fecha_hora'].dt.hour
    df['minuto'] = df['fecha_hora'].dt.minute
    return df

df = cargar_datos()

# --- Título de la app ---
st.title("🎯 Predicción Italia Keno por Hora y Minuto")

# --- Entradas de usuario ---
hora = st.selectbox("Selecciona la hora (0 a 23)", options=list(range(0, 24)), index=17)
minuto = st.selectbox("Selecciona los minutos (cada 5 min)", options=list(range(0, 60, 5)), index=6)

# --- Derivar features para predicción ---
dia_semana = pd.Timestamp.now().dayofweek
cuartil_hora = pd.cut([hora], bins=[-1, 5, 11, 17, 23], labels=[0, 1, 2, 3])[0]
minuto_5 = minuto // 5

# --- Preparar input ---
df_input = pd.DataFrame([{
    'hora': hora,
    'minuto': minuto,
    'dia_semana': dia_semana,
    'minuto_5': minuto_5,
    'cuartil_hora_1': int(cuartil_hora == 1),
    'cuartil_hora_2': int(cuartil_hora == 2),
    'cuartil_hora_3': int(cuartil_hora == 3)
}])

# --- Escalar y predecir ---
X_scaled = scaler.transform(df_input)
proba = model.predict_proba(X_scaled)[0]
top3_idx = np.argsort(proba)[-3:][::-1]
top3_numeros = le.inverse_transform(top3_idx)
top3_probs = proba[top3_idx]

# --- Mostrar predicción del modelo ---
st.subheader("🔮 Top 3 Números más Probables:")
for i, (num, prob) in enumerate(zip(top3_numeros, top3_probs), 1):
    st.write(f"**{i}. Número {num}** - Probabilidad: {prob:.2%}")

# --- Mostrar número más frecuente históricamente en esa hora y minuto ---
df_filtrado = df[(df['hora'] == hora) & (df['minuto'] == minuto)]

if not df_filtrado.empty:
    columnas_numeros = [f'n{i}' for i in range(1, 21)]
    todos = pd.Series(df_filtrado[columnas_numeros].values.ravel())
    top_historico = todos.value_counts().idxmax()
    freq = todos.value_counts().max()

    st.subheader("📊 Históricamente en esa hora y minuto:")
    st.success(f"El número que más veces ha salido es el **{top_historico}**, con **{freq} apariciones**.")
else:
    st.warning("⚠️ No hay registros suficientes para esa hora y minuto.")
