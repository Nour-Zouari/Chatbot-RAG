# ------------------------------
# IMPORTATIONS ET CONFIGURATION DOTENV
# ------------------------------
import os
import glob
import requests
import psycopg
import numpy as np
from psycopg import Cursor
from typing import List
from dotenv import load_dotenv 

# Charge les variables d'environnement
load_dotenv() 

# ------------------------------
# CONSTANTES ET VARIABLES D'ENVIRONNEMENT (Inchangées)
# ------------------------------
EMBEDDING_DIMENSION = 768 # (768 pour le modèle text-embedding-004)
TOP_K = 5 # nombre d'embeddings à chercher pour une question donnée
TXT_FOLDER_PATH = "data/TRANS_TXT" 

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "text-embedding-004") 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
GEMINI_GENERATION_MODEL = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")

DB_CONNECTION_STR = os.getenv(
    "DB_CONNECTION_STR",
    "dbname=chatbot user=postgres password=1234 host=127.0.0.1 port=5433"
) # les params pour la connexion à la base de données 

# ------------------------------
# FONCTION UTILITAIRE : VÉRIFIER L'INDEXATION
# ------------------------------
def is_corpus_indexed(cursor: Cursor) -> bool:
    """Vérifie si la table d'embeddings contient déjà des données."""
    try:
        # Compte le nombre de lignes dans la table
        cursor.execute("SELECT COUNT(*) FROM embeddings_gemini;")
        count = cursor.fetchone()[0]
        # On suppose qu'un corpus est indexé s'il y a plus de 81 passages 
        # (on a 41 fichiers chacun comportant au moins 2 passages)
        return count > 81
    except psycopg.Error as e:
        print(f"[ERREUR DB] Impossible de vérifier l'indexation: {e}")
        # En cas d'erreur (ex: table non créée), on renvoie False pour forcer l'ingestion.
        return False

# ------------------------------
# FONCTIONS RAG 
# ------------------------------

def parse_txt_file(file_path: str) -> List[str]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="cp1252") as f: #pour les accents et les caractères spéciaux
            text = f.read()
    
    lines = text.split("\n")
    passages = [line.removeprefix("    ") for line in lines if line.strip() and not line.startswith("<")] 
    return passages

def calculate_embeddings(text: str) -> List[float]:
    if not GEMINI_API_KEY:
        print(f"[WARNING] GEMINI_API_KEY non définie, utilisation d'embeddings factices ({EMBEDDING_DIMENSION} dim)")
        return np.random.rand(EMBEDDING_DIMENSION).tolist()
    
    EMBEDDING_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:embedContent"
    params = {"key": GEMINI_API_KEY}
    data = {"content": {"parts": [{"text": text}]}}

    try:
        response = requests.post(EMBEDDING_URL, json=data, params=params, timeout=10)
        response.raise_for_status() 
        resp_json = response.json()
        embedding_values = resp_json.get("embedding", {}).get("values")
        if not embedding_values:
            raise ValueError("Champ d'embedding 'values' manquant dans la réponse de l'API.")
        return embedding_values
        
    except requests.exceptions.RequestException as e:
        print(f"[WARNING] Erreur HTTP ou Réseau lors de l'appel à l'API Gemini ({e})")
    except Exception as e:
        print(f"[WARNING] Erreur de traitement de la réponse de l'API Gemini ou ValueError ({e})")
        
    return np.random.rand(EMBEDDING_DIMENSION).tolist()


def save_embedding(corpus: str, conversation_id: int, embedding: List[float], cursor: Cursor) -> None:
    cursor.execute(
        """
        INSERT INTO embeddings_gemini (conversation_id, corpus, embedding)
        VALUES (%s, %s, %s)
        """,
        (conversation_id, corpus, embedding)
    )

def similar_corpus(input_text: str, cursor: Cursor, top_k: int = TOP_K) -> List[tuple[int, str]]:
    embedding = calculate_embeddings(input_text)
    cursor.execute(
        """
        SELECT id, corpus
        FROM embeddings_gemini
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (embedding, top_k)
    )
    return cursor.fetchall()


def build_prompt(similar_texts: List[tuple[int, str]], question: str) -> str:
    context = "\n".join([text for _, text in similar_texts])
    prompt = f"Voici les conversations similaires trouvées dans la base :\n---\n{context}\n---\n\n Répond à cette question en se basant sur les anciennes réponses de l'hotesse (comme si c'est l'hotesse qui répond) \n Question : {question}\nRéponse :"
    return prompt


def generate_response(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "[ERREUR] GEMINI_API_KEY non définie. Impossible d'appeler l'API de génération."
    
    GENERATION_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_GENERATION_MODEL}:generateContent"
    params = {"key": GEMINI_API_KEY}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(GENERATION_URL, json=data, params=params, timeout=30) 
        response.raise_for_status()
        resp_json = response.json()
        
        if resp_json.get("candidates"):
            return resp_json["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "[RÉPONSE VIDE] Le modèle n'a pas pu générer de réponse (peut-être bloqué par les filtres de sécurité)."

    except requests.exceptions.RequestException as e:
        return f"[ERREUR API DE GÉNÉRATION] Réseau ou HTTP: {e}"
    except Exception as e:
        return f"[ERREUR] Impossible de traiter la réponse du LLM: {e}"

# ------------------------------
# CONNEXION À LA BASE ET INITIALISATION (Logique Conditionnelle)
# ------------------------------
print(f"Connexion à la base de données : {DB_CONNECTION_STR}")
with psycopg.connect(DB_CONNECTION_STR) as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        # Création de l'extension et de la table (nécessaire à chaque fois si la DB est nouvelle)
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings_gemini (
                id SERIAL PRIMARY KEY,
                conversation_id INT,
                corpus TEXT,
                embedding VECTOR({EMBEDDING_DIMENSION})
            );
        """)
        
        # ------------------------------
        # LOGIQUE D'INGESTION CONDITIONNELLE
        # ------------------------------
        
        txt_files_to_process = glob.glob(os.path.join(TXT_FOLDER_PATH, "*.txt"))
        num_files = len(txt_files_to_process)
        
        if is_corpus_indexed(cur):
            print("Corpus déjà indexé. Démarrage rapide du Chatbot.")
        else:
            print("Corpus non indexé ou vidé. Début de l'indexation complète.")
            
            print(f"Début du traitement de {num_files} fichier(s)...")
            print("CETTE ÉTAPE PEUT PRENDRE UN CERTAIN TEMPS.")

            for conv_id, file_path in enumerate(txt_files_to_process, start=1):
                try:
                    passages = parse_txt_file(file_path)
                    
                    for passage in passages:
                        embedding = calculate_embeddings(passage)
                        if len(embedding) == EMBEDDING_DIMENSION:
                            save_embedding(passage, conv_id, embedding, cur)
                        # else:
                        #     print(f"    [ERREUR] Taille d'embedding incorrecte. Passage ignoré.")
                            
                except Exception as e:
                    print(f"|-- [ERREUR CRITIQUE] Impossible de traiter le fichier {file_path}: {e}")
            
            print(" Indexation Terminé. Le Chatbot est prêt.")


        # ------------------------------
        # BOUCLE INTERACTIVE DU CHATBOT RAG 
        # ------------------------------
        
        print("\n" + "="*50)
        print("CHATBOT RAG (PRÊT)")
        print(f"Corpus indexé : {num_files} fichiers.")
        print("Tapez 'quitter' pour terminer la session.")
        print("="*50)
        
        while True:
            # L'utilisateur entre sa question
            user_question = input("\n[VOUS] : ")
            
            # Condition de sortie
            if user_question.lower() in ["quitter", "exit"]:
                print("\nAu revoir ! Fermeture du chatbot.")
                break

            # S'assurer que la question n'est pas vide
            if not user_question.strip():
                continue

            # 1. RETRIEVAL (Recherche des passages pertinents)
            print("Recherche des passages pertinents...")
            top_passages = similar_corpus(user_question, cur)
            
            if top_passages:
                # Affichage des passages trouvés (pour débogage)
                print(f"   (Passages trouvés - Top {len(top_passages)}) :")
                for _, text in top_passages:
                    print(f"    - '{text[:50]}...'")
                
                # 2. AUGMENTATION (Construction du prompt)
                prompt = build_prompt(top_passages, user_question)
                
                # 3. GENERATION (Appel au LLM Gemini)
                print("Génération de la réponse (Appel Gemini)...")
                llm_response = generate_response(prompt) 
                
                print("\n[CHATBOT] :")
                print(llm_response)
                
            else:
                print("\n[CHATBOT] :")
                print("je n'ai trouvé aucune information pertinente dans ma base de données pour répondre à cette question.")