# Importations
import requests
import psycopg
from psycopg import Cursor

# Variables
conversation_file_path = "chemin/vers/ton/fichier.txt"
GROK_MODEL = "grok-embedding-1"
GROK_API_URL = "https://api.grok.ai/v1/embeddings"
GROK_API_KEY = "ton_token"
db_connection_str = "dbname=chatbot user=postgres password=1234 host=localhost port=5433"

# --------------------------------------
# Fonction pour lire et filtrer le fichier
def create_conversation_list(file_path: str) -> list[str]:
    with open(file_path, "r") as file:
        text = file.read()
        text_list = text.split("\n")
        filtered_list = [ligne.removeprefix("     ") for ligne in text_list if not ligne.startswith("<")]
        print(f"Liste des textes extraits: {filtered_list}")
        return filtered_list

# --------------------------------------
# Fonction pour calculer les embeddings via Grok
def calculate_embeddings(corpus: str) -> list[float]:
    headers = {"Authorization": f"Bearer {GROK_API_KEY}"}
    data = {"input": corpus, "model": GROK_MODEL}
    
    response = requests.post(GROK_API_URL, json=data, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Erreur API Grok: {response.status_code} - {response.text}")
    
    response_json = response.json()
    return response_json["embedding"]  # adapter la clé si nécessaire

# --------------------------------------
# Fonction pour insérer un embedding dans la table
def save_embedding(corpus: str, embedding: list[float], cursor: Cursor) -> None:
    cursor.execute(
        'INSERT INTO embeddings (corpus, embedding) VALUES (%s, %s)',
        (corpus, embedding)
    )

# --------------------------------------
# Fonction pour récupérer les textes similaires
def similar_corpus(input_corpus: str, cursor: Cursor) -> list[tuple[int, str]]:
    embedding = calculate_embeddings(input_corpus)
    # Avec FLOAT8[], pgvector n’est pas utilisé → recherche brute
    cursor.execute(
        "SELECT id, corpus FROM embeddings"  # ici, recherche brute, sans distance optimisée
    )
    results = cursor.fetchall()
    # Calcul manuel de distance si besoin (cosine, euclidean) peut être fait en Python
    return results

# --------------------------------------
# Connexion à la base et exécution
with psycopg.connect(db_connection_str) as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        # Supprime la table si elle existe
        cur.execute("DROP TABLE IF EXISTS embeddings;")
        
        # Créer la table embeddings avec FLOAT8[]
        cur.execute("""
            CREATE TABLE embeddings (
                id SERIAL PRIMARY KEY,
                corpus TEXT,
                embedding FLOAT8[]
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
        print("Textes dans la table :")
        for id_, text in similar_texts:
            print(f"{id_}: {text}")
