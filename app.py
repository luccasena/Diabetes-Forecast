# Importando as bibliotecas necessárias

import streamlit as st
import numpy as np
import joblib
from agent import agentDiagnosis

# Configurando os elementos da página do Streamlit
st.set_page_config(page_title="Predição de Diabetes", page_icon=":hospital:", layout="wide")

col1, col2, col3 = st.columns(3)

with col1:
    pass

with col2:
    st.image("img/EvoEndo.png" , width=400)

with col3:
    pass


st.title("Machine Learning e Agente de IA: Previsão de Diabetes")
st.markdown("Este aplicativo utiliza um modelo de Machine Learning para prever o risco de Diabetes e um agente de IA para fornecer recomendações médicas personalizadas com base nos dados do paciente.")
st.markdown("---")


col1, col2 = st.columns(2)

with col1:
    nome = st.text_input("Digite seu Nome Completo:", key="nome")

    idade = st.number_input("1. Digite sua Idade:", key="idade", format="%.0f")

    genero = st.selectbox("2. Digite seu gênero biológico", options=["Masculino", "Feminino"],key="genero")

    imc = st.number_input("3. Digite seu IMC", key="imc", format="%.2f")

    glicemia = st.number_input("4. Digite seu nível de Glicemia (mg/dL)", format="%.2f", key="glicemia")

    insulina = st.number_input("6. Digite sua insulina (μU/mL)", format="%.2f", key="insulina")



with col2:
    hba1c = st.number_input("5. Digite sua Hemoglobina Glicada (%)", format="%.0f", key="hba1c")

    historico_familiar = st.selectbox("8. Digite se tem hístorico familiar", options=["Sim", "Não"], key="historico_familiar")

    pressao_arterial = st.number_input("9. Digite sua pressao arterial (mmHg)", format="%.2f", key="pressao_arterial")

    colesterol_total = st.number_input("10. Digite sua colesterol total (mg/dL)", format="%.2f", key="colesterol_total")

    triglicerideos = st.number_input("11. Digite sua triglicerideos (mg/dL)", format="%.2f", key="triglicerideos")

    atividade_fisica = st.slider("7. Digite seu nível de Atividade Física",key="atividade_fisica", min_value=0, max_value=9)



user_info = np.array([idade,genero,imc,glicemia,hba1c,insulina,atividade_fisica,historico_familiar,pressao_arterial,colesterol_total,triglicerideos])

st.markdown("---")

if st.button("Gerar Previsão e Relatório", key="predict_button"):

    if len(user_info) == 11:

        # Carregando o modelo e o pré-processador:
        random_forest = joblib.load("ml_models/modelo_random_forest.pkl")
        transformer = joblib.load("ml_models/transformer.pkl")

        # Tratando os dados de entrada:
        historico_familiar = 1 if historico_familiar == "Sim" else 0

        # Transformando os dados de entrada em um array numpy:
        user_info_clean = np.array([idade,genero,imc,glicemia,hba1c,insulina,atividade_fisica,historico_familiar,pressao_arterial,colesterol_total,triglicerideos])

        # Transformando os dados de entrada usando o pré-processador:
        user_info_transformer = transformer.transform([user_info_clean])

        # Fazendo a previsão:
        prediction = random_forest.predict(user_info_transformer)
        prediction_proba = random_forest.predict_proba(user_info_transformer)

        st.session_state['prediction'] = prediction
        st.session_state['user_info'] = user_info

        if prediction == 'Não diabético':
            proba = prediction_proba[0][2]
            st.session_state['proba'] = proba

        elif prediction == 'Diabetes Tipo 1':
            proba = prediction_proba[0][0]
            st.session_state['proba'] = proba                

        elif prediction == 'Pré-diabetes':
            proba = prediction_proba[0][3]
            st.session_state['proba'] = proba                

        elif prediction == 'Diabetes Tipo 2':
            proba = prediction_proba[0][1]
            st.session_state['proba'] = proba                

        report = agentDiagnosis(
        nome,
        st.session_state['user_info'],
        st.session_state['prediction'],
        st.session_state['proba']
    )
        
        st.markdown("---")
        st.subheader("Relatório do Agente de Diagnóstico Médico:")
        st.session_state.relatorio_gerado = False
        st.markdown(report)

    else:
        st.warning("Por favor, preencha todos os campos antes de fazer a previsão.")

