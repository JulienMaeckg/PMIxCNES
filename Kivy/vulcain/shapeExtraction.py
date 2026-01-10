from PIL import Image
import numpy as np
import requests
import io


def load_full_resolution_image(path):
    """Charge l'image directement sans gestion MPF."""
    print(f"[SHAPE] Chargement simple de: {path}")
    img = Image.open(path)
    img.load()
    print(f"[SHAPE] Image chargée: {img.size}, mode: {img.mode}")
    return img


def resize_image(img, max_size=2000):
    """Redimensionne proprement l'image avant l'envoi pour limiter la bande passante."""
    
    # --- CORRECTION MAJEURE ---
    # On force la conversion en RGB via Pillow dès le début.
    # Cela corrige le bug où les photos Android étaient vues comme des matrices 
    # de gris (2D) ou des palettes (P) par NumPy.
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Conversion de l'objet PIL en tableau NumPy
    img_array = np.array(img)

    # Note : Le bloc "if len(img_array.shape) == 2" est supprimé car
    # img.convert('RGB') garantit que nous avons maintenant 3 dimensions (Couleurs).

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
            print(f"[SHAPE] Image PIL reçue: {img.size}, mode: {img.mode}")
            img_to_send = resize_image(img)
        elif isinstance(img, str):
            print(f"[SHAPE] Chemin reçu: {img}")
            img_loaded = load_full_resolution_image(img)
            print(f"[SHAPE] Image chargée: {img_loaded.size}, mode: {img_loaded.mode}")
            img_to_send = resize_image(img_loaded)
        else:
            raise ValueError("img doit être une Image PIL ou un chemin de fichier")

        print(f"[SHAPE] Image après resize: {img_to_send.size}")

        # Conversion de l'image en flux binaire (mémoire vive) au format JPEG
        img_bytes = io.BytesIO()
        img_to_send.save(img_bytes, format="JPEG", quality=95)
        img_bytes.seek(0)  # Remise à zéro du curseur de lecture

        print(f"[SHAPE] Taille du fichier à envoyer: {len(img_bytes.getvalue())} bytes")

        # Préparation du dictionnaire de fichiers pour la requête HTTP POST
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}

        # Envoi effectif au serveur avec un délai d'attente de 60 secondes
        response = requests.post(url, files=files, timeout=60)

        # Si le serveur répond avec succès (Code 200)
        if response.status_code == 200:
            data = response.json()

            print("[SHAPE] Données reçues du serveur avec succès")
            
            return data
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