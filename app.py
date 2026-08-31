
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = "models"

st.set_page_config(page_title="Customer Support Ticket System", page_icon="🎫")


@st.cache_resource
def load_artifacts():
    intent_vectorizer = joblib.load(f"{MODELS_DIR}/intent_vectorizer.joblib")
    svm_model = joblib.load(f"{MODELS_DIR}/intent_model.joblib")
    priority_vectorizer = joblib.load(f"{MODELS_DIR}/priority_vectorizer.joblib")
    svm_priority_model = joblib.load(f"{MODELS_DIR}/priority_model.joblib")
    retrieval_vectorizer = joblib.load(f"{MODELS_DIR}/retrieval_vectorizer.joblib")
    instruction_vectors = joblib.load(f"{MODELS_DIR}/instruction_vectors.joblib")
    raw_df = pd.read_csv(f"{MODELS_DIR}/responses.csv")
    return (
        intent_vectorizer,
        svm_model,
        priority_vectorizer,
        svm_priority_model,
        retrieval_vectorizer,
        instruction_vectors,
        raw_df,
    )


(
    intent_vectorizer,
    svm_model,
    priority_vectorizer,
    svm_priority_model,
    retrieval_vectorizer,
    instruction_vectors,
    raw_df,
) = load_artifacts()


def retrieve_response(user_query):
    query_vector = retrieval_vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(query_vector, instruction_vectors)
    best_match = similarity_scores.argmax()
    return raw_df.iloc[best_match]["response"]


def customer_support_system(user_query):
    intent_query = intent_vectorizer.transform([user_query])
    predicted_intent = svm_model.predict(intent_query)[0]

    priority_query = priority_vectorizer.transform([user_query])
    predicted_priority = svm_priority_model.predict(priority_query)[0]

    response = retrieve_response(user_query)

    return predicted_intent, predicted_priority, response


st.title("🎫 Customer Support Ticket System")

user_query = st.text_input("Customer Query:", placeholder="e.g. I forgot my password")

if st.button("Submit") and user_query.strip():
    predicted_intent, predicted_priority, response = customer_support_system(user_query)

    st.write("**Predicted Intent:**", predicted_intent)
    st.write("**Predicted Priority:**", predicted_priority)
    st.write("**Retrieved Response:**")
    st.info(response)
