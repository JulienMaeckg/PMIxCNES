from PIL import Image
import numpy as np
import requests
import io


def resize_image(img, max_size=2000):
    """Redimensionne proprement l'image avant l'envoi pour limiter la bande passante."""
    # Conversion de l'objet PIL en tableau NumPy
    img_array = np.array(img)

    # Conversion forcée en RGB si l'image est en niveaux de gris (2 dimensions)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    # Suppression du canal de transparence (Alpha) si présent pour compatibilité JPEG
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    # Récupération des dimensions actuelles
    height, width = img_array.shape[:2]

    # Calcul des nouvelles dimensions si le côté le plus grand dépasse max_size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        # Reconversion en objet PIL pour utiliser l'algorithme LANCZOS (haute qualité)
        img_pil_temp = Image.fromarray(img_array)

        # Gestion de la compatibilité selon la version de Pillow installée
        try:
            img_resized = img_pil_temp.resize((new_width, new_height), Image.LANCZOS)
        except AttributeError:
            img_resized = img_pil_temp.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )

        return img_resized
    else:
        # Retourne l'image telle quelle (en format PIL propre) si elle est déjà petite
        return Image.fromarray(img_array)


def dimensions(img):
    """Envoie l'image au serveur FastAPI et extrait les dimensions calculées."""
    # Adresse IP du serveur distant réalisant le traitement IA
    url = "http://51.178.50.148:8000/predict"

    try:
        # Préparation de l'image (Chargement et Redimensionnement)
        if isinstance(img, Image.Image):
            img_to_send = resize_image(img)
        elif isinstance(img, str):
            img_to_send = resize_image(Image.open(img))
        else:
            raise ValueError("img doit être une Image PIL ou un chemin de fichier")

        # Conversion de l'image en flux binaire (mémoire vive) au format JPEG
        img_bytes = io.BytesIO()
        img_to_send.save(img_bytes, format="JPEG")
        img_bytes.seek(0)  # Remise à zéro du curseur de lecture
        # Préparation du dictionnaire de fichiers pour la requête HTTP POST
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}

        # Envoi effectif au serveur avec un délai d'attente de 60 secondes
        response = requests.post(url, files=files, timeout=60)

        # Si le serveur répond avec succès (Code 200)
        if response.status_code == 200:
            data = response.json()

            # Extraction de la matrice de l'ogive (list) pour la traiter séparément
            img_ogive_list = data.pop("img_ogive")
            # Nettoyage des métadonnées de forme inutiles ici
            data.pop("img_ogive_shape", None)

            # Stockage des dimensions extraites (le dictionnaire principal)
            Donnes = data

            # Reconversion de la liste JSON en véritable matrice numérique NumPy
            img_ogive = np.array(img_ogive_list)

            print("[SHAPE] Données et matrice reçues du serveur avec succès")
            # Retourne les deux éléments nécessaires à la suite du calcul
            return Donnes["donnees"], img_ogive
        else:
            # En cas d'erreur serveur (ex: 500), affiche le message d'erreur
            print(f"[SHAPE] Erreur {response.status_code} : {response.text}")
            raise Exception(f"Erreur serveur: {response.status_code}")

    except requests.exceptions.Timeout:
        # Gestion spécifique du cas où le serveur est trop lent
        print("[SHAPE] Timeout - Le serveur met trop de temps à répondre")
        raise Exception("Timeout lors de la connexion au serveur")
    except requests.exceptions.ConnectionError:
        # Gestion du cas où le serveur est hors-ligne ou inaccessible
        print("[SHAPE] Erreur de connexion - Impossible de joindre le serveur")
        raise Exception("Impossible de se connecter au serveur")
    except Exception as e:
        # Capture toutes les autres erreurs imprévues
        print(f"[SHAPE] Erreur lors de l'envoi : {e}")
        raise