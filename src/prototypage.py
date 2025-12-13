# ------------------------------
# IMPORTATIONS
# ------------------------------
import os
import glob
import requests
import psycopg
from psycopg import Cursor

# ------------------------------
# VARIABLES D'ENVIRONNEMENT (sécurisées)
# ------------------------------
GROK_MODEL = os.getenv("GROK_MODEL", "grok-embedding-1")
GROK_API_KEY = os.getenv("GROK_API_KEY")  #  .env 
DB_CONNECTION_STR = os.getenv(
    "DB_CONNECTION_STR",
    "dbname=chatbot user=postgres password=1234 host=localhost port=5433"
)
TOP_K = 5  # nombre de passages récupérés

# ------------------------------
# FONCTION : LIRE UN FICHIER TXT ET NETTOYER
# ------------------------------
def parse_txt_file(file_path: str) -> list[str]:
    """
    Lit un fichier txt et renvoie une liste de passages.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
        passages = [line.removeprefix("     ") for line in lines if line.strip() ]
    return passages

# ------------------------------
# FONCTION : CALCUL EMBEDDING VIA GROK
# ------------------------------
def calculate_embeddings(text: str) -> list[float]:
    """
    Envoie le texte à l'API Grok pour obtenir l'embedding.
    """
    headers = {"Authorization": f"Bearer {GROK_API_KEY}"}
    data = {"input": text, "model": GROK_MODEL}
    response = requests.post("https://api.grok.ai/v1/embeddings", json=data, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Erreur API Grok: {response.status_code} - {response.text}")
    
    resp_json = response.json()
    # Adapter selon la structure exacte renvoyée par Grok
    return resp_json["embedding"]  # ou resp_json["data"][0]["embedding"]

# ------------------------------
# FONCTION : INSERTION D'UN PASSAGE
# ------------------------------
def save_embedding(corpus: str, conversation_id: int, embedding: list[float], cursor: Cursor) -> None:
    """
    Insère un passage avec son embedding et son conversation_id dans la table embeddings.
    """
    cursor.execute(
        "INSERT INTO embeddings (conversation_id, corpus, embedding) VALUES (%s, %s, %s)",
        (conversation_id, corpus, embedding)
    )

# ------------------------------
# FONCTION : RECHERCHE TOP-K PASSAGES
# ------------------------------
def similar_corpus(input_text: str, cursor: Cursor, top_k: int = TOP_K) -> list[tuple[int, str]]:
    """
    Calcule l'embedding de la question et renvoie les top-k passages les plus proches.
    """
    embedding = calculate_embeddings(input_text)
    cursor.execute(
        "SELECT id, corpus FROM embeddings ORDER BY embedding <=> %s LIMIT %s",
        (embedding, top_k)
    )
    return cursor.fetchall()

# ------------------------------
# FONCTION : CONSTRUCTION DU PROMPT POUR LE LLM
# ------------------------------
def build_prompt(similar_texts: list[tuple[int, str]], question: str) -> str:
    """
    Construit le contexte et la question pour le LLM.
    """
    context = "\n".join([text for _, text in similar_texts])
    prompt = f"Voici le contexte extrait de la base :\n{context}\n\nQuestion : {question}\nRéponse :"
    return prompt

# ------------------------------
# FONCTION : SIMULATION D'APPEL AU LLM
# ------------------------------
def call_llm(prompt: str) -> str:
    """
    Simule l'appel au LLM. Ici tu peux brancher OpenAI, Gemini, Ollama, etc.
    """
    # Exemple conceptuel : renvoie le prompt pour test
    # Remplace par LLM.generate(prompt) réel
    return f"[Réponse générée par le LLM pour le prompt donné]\n{prompt}"

# ------------------------------
# CONNEXION À LA BASE ET INITIALISATION
# ------------------------------
with psycopg.connect(DB_CONNECTION_STR) as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        # Activation de l'extension vectorielle
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Création de la table embeddings (si inexistante)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                conversation_id INT,
                corpus TEXT,
                embedding VECTOR(1024)
            );
        """)
        
        # ------------------------------
        # TRAITEMENT DE TOUS LES FICHIERS TXT DANS UN DOSSIER
        # ------------------------------
        txt_files = glob.glob("data/TRANS_TXT/*.txt")  # dossier contenant tous les fichiers
        for conv_id, file_path in enumerate(txt_files, start=1):
            passages = parse_txt_file(file_path)
            for passage in passages:
                embedding = calculate_embeddings(passage)
                save_embedding(passage, conv_id, embedding, cur)
        
        # ------------------------------
        # EXEMPLE D'INTERROGATION ET GENERATION DE REPONSE
        # ------------------------------
        user_question = "Exemple de question à poser au chatbot RAG"
        top_passages = similar_corpus(user_question, cur)
        prompt = build_prompt(top_passages, user_question)
        llm_response = call_llm(prompt)
        print("Réponse du Chatbot RAG :\n", llm_response)
