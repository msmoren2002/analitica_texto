import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk
from transformers import pipeline

# Descargar stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
import re

# -------------------------------
# Funciones auxiliares
# -------------------------------

def limpiar_texto(texto):
    texto = re.sub(r"http\S+|www\S+|https\S+", '', texto, flags=re.MULTILINE)
    texto = re.sub(r'\@w+|\#','', texto)
    texto = re.sub(r'[^A-Za-zÀ-ÿñÑ ]+', '', texto)
    texto = texto.lower()
    return texto

def eliminar_stopwords(texto):
    stop_words = set(stopwords.words('spanish'))
    palabras = texto.split()
    palabras_filtradas = [p for p in palabras if p not in stop_words]
    return " ".join(palabras_filtradas)

def generar_nube_palabras(textos):
    texto_total = " ".join(textos)
    wc = WordCloud(width=800, height=400, background_color='white').generate(texto_total)
    st.image(wc.to_array())

def palabras_mas_frecuentes(textos):
    palabras = " ".join(textos).split()
    counter = Counter(palabras)
    comunes = counter.most_common(10)
    palabras, frecuencias = zip(*comunes)
    fig, ax = plt.subplots()
    ax.bar(palabras, frecuencias)
    ax.set_xticklabels(palabras, rotation=45)
    fig.tight_layout()
    plt.close(fig)
    st.pyplot(fig)

# -------------------------------
# Interfaz Streamlit
# -------------------------------

st.title("Análisis de Opiniones de Clientes")

archivo = st.file_uploader("Sube un archivo CSV con opiniones", type=["csv"])

if not archivo:
    st.info("Por favor, sube un archivo CSV para comenzar.")
    st.stop()

df = pd.read_csv(archivo)
st.write("Vista previa de los datos:", df.head())

if 'texto' not in df.columns:
    st.warning("El archivo debe contener una columna llamada 'texto'.")
    st.stop()

# Procesamiento de texto
st.subheader("Procesamiento de texto")
with st.spinner("Limpiando y procesando opiniones..."):
    df['texto_limpio'] = df['texto'].astype(str).apply(limpiar_texto).apply(eliminar_stopwords)
st.success("Opiniones procesadas.")

# Nube de palabras
st.subheader("Nube de palabras")
generar_nube_palabras(df['texto_limpio'])

# Palabras más frecuentes
st.subheader("Top 10 palabras más frecuentes")
palabras_mas_frecuentes(df['texto_limpio'])

# Clasificación de sentimientos
st.subheader("Clasificación de sentimientos")
with st.spinner("Cargando modelo de clasificación de sentimientos..."):
    try:
        clasificador = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
        df['sentimiento'] = df['texto'].apply(lambda x: clasificador(x[:512])[0]['label'])
        st.success("Clasificación completada.")
    except Exception as e:
        st.error(f"No se pudo cargar el modelo de sentimiento: {e}")
        st.stop()

st.write(df[['texto', 'sentimiento']])

# Gráfico de sentimientos
st.subheader("Distribución de sentimientos")
conteo = df['sentimiento'].value_counts()
st.bar_chart(conteo)

# Consulta sobre opiniones
st.subheader("Consulta sobre las opiniones")
pregunta = st.text_input("Haz una pregunta sobre las opiniones")

if pregunta:
    with st.spinner("Buscando respuesta..."):
        try:
            qa = pipeline("question-answering")
            contexto = " ".join(df['texto'].tolist())[:1000]
            respuesta = qa(question=pregunta, context=contexto)
            st.write("Respuesta:", respuesta['answer'])
        except Exception as e:
            st.error(f"No se pudo generar respuesta: {e}")
