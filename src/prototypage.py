# Importations nécessaires
import requests
import psycopg
from psycopg import Cursor

# Variables
conversation_file_path = "chemin/vers/ton/fichier.txt"
# modèle Grok pour les embeddings
GROK_MODEL = "grok-embedding-1"
GROK_API_URL = "https://api.grok.ai/v1/embeddings"
GROK_API_KEY = "ton_token"

# Connexion à PostgreSQL
db_connection_str = "dbname=chatbot user=postgres password=1234 host=localhost port=5433"

# --------------------------------------
# Fonction pour lire et filtrer le fichier de conversation
def create_conversation_list(file_path: str) -> list[str]:
    """
    Lit le fichier ligne par ligne, ignore les lignes commençant par '<'
    et supprime les espaces de début.
    """
    with open(file_path, "r") as file:
        text = file.read()
        text_list = text.split("\n")
        filtered_list = [ligne.removeprefix("     ") for ligne in text_list if not ligne.startswith("<")]
        print(f"Liste des textes extraits: {filtered_list}")
        return filtered_list

# --------------------------------------
# Fonction pour calculer les embeddings via Grok
def calculate_embeddings(corpus: str) -> list[float]:
    """
    Envoie le texte à l'API Grok et récupère l'embedding correspondant.
    """
    headers = {"Authorization": f"Bearer {GROK_API_KEY}"}
    data = {"input": corpus, "model": GROK_MODEL}
    
    response = requests.post(GROK_API_URL, json=data, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Erreur API Grok: {response.status_code} - {response.text}")
    
    response_json = response.json()
    # Grok retourne normalement le vecteur sous "embedding" ou "data[0].embedding"
    return response_json["embedding"]  # adapter si nécessaire

# --------------------------------------
# Fonction pour insérer un embedding dans la base
def save_embedding(corpus: str, embedding: list[float], cursor: Cursor) -> None:
    """
    Insère un texte et son embedding dans la table embeddings.
    """
    cursor.execute(
        'INSERT INTO embeddings (corpus, embedding) VALUES (%s, %s)',
        (corpus, embedding)
    )

# --------------------------------------
# Fonction pour rechercher les textes similaires
def similar_corpus(input_corpus: str, cursor: Cursor) -> list[tuple[int, str]]:
    """
    Calcule l'embedding du texte d'entrée et renvoie les 5 textes les plus proches
    selon la distance cosinus (opérateur <=> de pgvector).
    """
    embedding = calculate_embeddings(input_corpus)
    cursor.execute(
        "SELECT id, corpus FROM embeddings ORDER BY embedding <=> %s LIMIT 5",
        (embedding,)
    )
    return cursor.fetchall()

# --------------------------------------
# Connexion à la base et exécution des opérations
with psycopg.connect(db_connection_str) as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        # au cas ou j'ai lancé le script plusieurs fois et embeddings existe déjà
        cur.execute("DROP TABLE IF EXISTS embeddings;")
        
        # j'ai ajouté IF NOT EXIST au cas ou je vais relancer le script plusieurs fois
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Créer la table embeddings avec type VECTOR
        # 1024 correspond à la dimension du vecteur renvoyé par le modèle Grok
        cur.execute("""
            CREATE TABLE embeddings (
                id SERIAL PRIMARY KEY,
                corpus TEXT,
                embedding VECTOR(1024)
            );
        """)
        
        # Lire le fichier et insérer les embeddings
        corpus_list = create_conversation_list(file_path=conversation_file_path)
        for corpus in corpus_list:
            embedding = calculate_embeddings(corpus)
            save_embedding(corpus, embedding, cur)
        
        conn.commit()
        
        # Exemple d'interrogation
        test_text = "Exemple de texte pour tester la similarité"
        similar_texts = similar_corpus(test_text, cur)
        print("Textes similaires :")
        for id_, text in similar_texts:
            print(f"{id_}: {text}")
