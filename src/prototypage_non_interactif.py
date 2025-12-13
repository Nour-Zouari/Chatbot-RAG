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
# Nécessaire pour charger les variables du fichier .env
from dotenv import load_dotenv 

# Charge les variables d'environnement
load_dotenv() 

# ------------------------------
# CONSTANTES ET VARIABLES D'ENVIRONNEMENT
# ------------------------------
# Constantes
EMBEDDING_DIMENSION = 768 # Dimension de sortie pour 'text-embedding-004'
TOP_K = 5
# MODIFICATION CLÉ: Pointez vers le DOSSIER contenant tous les .txt pour l'indexation
TXT_FOLDER_PATH = "data/TRANS_TXT" 

# Variables d'environnement chargées via os.getenv
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "text-embedding-004") 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Lecture de la clé depuis l'environnement (ex: .env)
GEMINI_GENERATION_MODEL = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash")

DB_CONNECTION_STR = os.getenv(
    "DB_CONNECTION_STR",
    "dbname=chatbot user=postgres password=1234 host=127.0.0.1 port=5433"
)

# ------------------------------
# FONCTION : LIRE UN FICHIER TXT AVEC ENCODAGE ROBUSTE
# ------------------------------
def parse_txt_file(file_path: str) -> List[str]:
    """Lit un fichier TXT avec gestion robuste des encodages et extrait les passages."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        # Tente l'encodage Windows par défaut si UTF-8 échoue
        with open(file_path, "r", encoding="cp1252") as f:
            text = f.read()
    
    lines = text.split("\n")
    # Retire le préfixe '    ' (4 espaces) et filtre les lignes vides ou de commentaires (<...)
    passages = [line.removeprefix("    ") for line in lines if line.strip() and not line.startswith("<")]
    return passages

# ------------------------------
# FONCTION : CALCUL EMBEDDINGS (GEMINI OU FACTICE)
# ------------------------------
def calculate_embeddings(text: str) -> List[float]:
    """Calcule l'embedding via l'API Gemini (text-embedding-004) ou renvoie un embedding factice."""
    if not GEMINI_API_KEY:
        print(f"[WARNING] GEMINI_API_KEY non définie, utilisation d'embeddings factices ({EMBEDDING_DIMENSION} dim)")
        return np.random.rand(EMBEDDING_DIMENSION).tolist()
    
    # URL OFFICIELLE pour l'embedding de contenu
    EMBEDDING_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:embedContent"

    # La clé API est passée en tant que paramètre de requête
    params = {"key": GEMINI_API_KEY}
    
    # Structure du corps de la requête attendue par 'embedContent'
    data = {
        "content": {"parts": [{"text": text}]},
    }

    try:
        response = requests.post(EMBEDDING_URL, json=data, params=params, timeout=10)
        response.raise_for_status() 
        
        resp_json = response.json()
        
        # Accès à l'embedding dans la structure de réponse standard
        embedding_values = resp_json.get("embedding", {}).get("values")
        
        if not embedding_values:
            raise ValueError("Champ d'embedding 'values' manquant dans la réponse de l'API.")

        return embedding_values
        
    except requests.exceptions.RequestException as e:
        print(f"[WARNING] Erreur HTTP ou Réseau lors de l'appel à l'API Gemini ({e}), utilisation d'embedding factice")
    except Exception as e:
        print(f"[WARNING] Erreur de traitement de la réponse de l'API Gemini ou ValueError ({e}), utilisation d'embedding factice")
        
    return np.random.rand(EMBEDDING_DIMENSION).tolist()

# ------------------------------
# FONCTION : INSERTION D'UN PASSAGE
# ------------------------------
def save_embedding(corpus: str, conversation_id: int, embedding: List[float], cursor: Cursor) -> None:
    """Insère un corpus et son embedding dans la base de données."""
    cursor.execute(
        """
        INSERT INTO embeddings_gemini (conversation_id, corpus, embedding)
        VALUES (%s, %s, %s)
        """,
        (conversation_id, corpus, embedding)
    )

# ------------------------------
# FONCTION : RECHERCHE TOP-K PASSAGES (RETRIEVAL)
# ------------------------------
def similar_corpus(input_text: str, cursor: Cursor, top_k: int = TOP_K) -> List[tuple[int, str]]:
    """Recherche les 'top_k' corpus les plus similaires à 'input_text'."""
    # 1. Calcul de l'embedding de la requête
    embedding = calculate_embeddings(input_text)
    
    # 2. Recherche par distance cosinus (opérateur <=> de pgvector)
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

# ------------------------------
# FONCTION : CONSTRUCTION DU PROMPT POUR LE LLM
# ------------------------------
def build_prompt(similar_texts: List[tuple[int, str]], question: str) -> str:
    """Construit le prompt RAG pour le LLM en utilisant le contexte récupéré."""
    # Extrait uniquement le texte des tuples (id, texte)
    context = "\n".join([text for _, text in similar_texts])
    prompt = f"Voici le contexte extrait de la base :\n---\n{context}\n---\n\nQuestion : {question}\nRéponse :"
    return prompt

# ------------------------------
# FONCTION : APPEL AU LLM (RÉEL)
# ------------------------------
def generate_response(prompt: str) -> str:
    """Appelle l'API de génération de Gemini pour obtenir la réponse finale."""
    if not GEMINI_API_KEY:
        return "[ERREUR] GEMINI_API_KEY non définie. Impossible d'appeler l'API de génération."
    
    GENERATION_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_GENERATION_MODEL}:generateContent"

    params = {"key": GEMINI_API_KEY}
    
    # Structure de la requête pour generateContent
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
    }

    try:
        # NOTE : Timeout augmenté pour la génération qui peut être plus longue
        response = requests.post(GENERATION_URL, json=data, params=params, timeout=30) 
        response.raise_for_status()
        resp_json = response.json()
        
        # Extrait le texte généré (avec vérification de structure)
        if resp_json.get("candidates"):
            return resp_json["candidates"][0]["content"]["parts"][0]["text"]
        else:
             # Peut arriver si la réponse est bloquée ou vide
            return "[RÉPONSE VIDE] Le modèle n'a pas pu générer de réponse (peut-être bloqué par les filtres de sécurité)."

    except requests.exceptions.RequestException as e:
        return f"[ERREUR API DE GÉNÉRATION] Réseau ou HTTP: {e}"
    except Exception as e:
        return f"[ERREUR] Impossible de traiter la réponse du LLM: {e}"
        
# ------------------------------
# CONNEXION À LA BASE ET INITIALISATION
# ------------------------------
print(f"Connexion à la base de données : {DB_CONNECTION_STR}")
with psycopg.connect(DB_CONNECTION_STR) as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        # Configuration de la base de données
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Création/vérification de la table avec la dimension correcte
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings_gemini (
                id SERIAL PRIMARY KEY,
                conversation_id INT,
                corpus TEXT,
                embedding VECTOR({EMBEDDING_DIMENSION})
            );
        """)
        
        # Purge des anciennes données pour le nouvel index complet
        print("Purge des anciennes données d'embeddings...")
        cur.execute("TRUNCATE TABLE embeddings_gemini RESTART IDENTITY;") 

        # ------------------------------
        # TRAITEMENT DE TOUS LES FICHIERS TXT
        # ------------------------------
        
        # MODIFICATION CLÉ : Utilisation de TXT_FOLDER_PATH pour tous les fichiers
        txt_files_to_process = glob.glob(os.path.join(TXT_FOLDER_PATH, "*.txt"))
        print(f"Début du traitement de {len(txt_files_to_process)} fichier(s)...")
        print("CETTE ÉTAPE PEUT PRENDRE PLUSIEURS MINUTES EN FONCTION DU NOMBRE DE FICHIERS.")

        for conv_id, file_path in enumerate(txt_files_to_process, start=1):
            try:
                passages = parse_txt_file(file_path)
                print(f"|-- Fichier {file_path}: {len(passages)} passages extraits")
                
                for passage in passages:
                    embedding = calculate_embeddings(passage)
                    if len(embedding) == EMBEDDING_DIMENSION:
                        save_embedding(passage, conv_id, embedding, cur)
                    else:
                        print(f"    [ERREUR] Taille d'embedding incorrecte ({len(embedding)} au lieu de {EMBEDDING_DIMENSION}). Passage ignoré.")
                        
            except Exception as e:
                print(f"|-- [ERREUR CRITIQUE] Impossible de traiter le fichier {file_path}: {e}")

        # ------------------------------
        # EXEMPLE DE QUESTION ET GENERATION DE REPONSE
        # ------------------------------
        print("\n" + "="*50)
        print(f"EXEMPLE DE RECHERCHE RAG (Corpus de {len(txt_files_to_process)} fichiers)")
        print("="*50)
        
        # Utilisation de la question plus spécifique pour tester la pertinence sur le grand corpus
        user_question = "bonjour j'aurais souhaité avoir le secrétariat de carrière juridique s'il vous plait"
        print(f"Question Utilisateur: {user_question}")
        
        top_passages = similar_corpus(user_question, cur)
        
        if top_passages:
            print(f"Passages Similaires Retrouvés (Top {len(top_passages)}) :")
            for _, text in top_passages:
                print(f" - '{text[:60]}...'")
            
            prompt = build_prompt(top_passages, user_question)
            
            # Utilisation de la fonction de génération réelle
            print("\nAppel de l'API de génération (Gemini)...")
            llm_response = generate_response(prompt) 
            
            print("\nRéponse du Chatbot RAG :")
            print(llm_response)
        else:
            print("Aucun passage similaire trouvé dans la base de données. Assurez-vous que l'indexation a réussi.")