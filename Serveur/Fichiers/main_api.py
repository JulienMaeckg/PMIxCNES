from fastapi import FastAPI, File, UploadFile
from datetime import datetime
from PIL import Image
import noseConeShape
import io
import os

from analyse_fusee import analyse_image  # ton code existant avec analyse_image()

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lire l'image envoyée par l'app
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes))

    # SAUVEGARDER L'IMAGE REÇUE
    save_dir = "/home/ubuntu/ml_server/received_images"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{save_dir}/photo_{timestamp}.jpg"
    image.save(save_path)
    print(f"[API] Image sauvegardée: {save_path}")
    
    print(f"[API] Taille fichier reçu: {len(img_bytes)} bytes")

    # --- ROTATION MANUELLE ---
    # On récupère largeur (w) et hauteur (h)
    w, h = image.size
    print(f"[API] Dimensions image: {w}x{h}")
    print(f"[API] Mode image: {image.mode}")
    
    # Si la largeur est supérieure à la hauteur, l'image est "couchée"
    if w > h:
        # On pivote de 90° vers la droite (ou -90 vers la gauche) 
        # pour la remettre debout
        image = image.rotate(-90, expand=True)
        print(f"[API] Image pivotée: {image.size}")
    # --------------------------

    # Appeler ton code d'analyse
    (
    hauteur_rouge,           # hauteur du fuselage
    epaisseur_moyenne_rouge, # épaisseur moyenne du fuselage
    hauteur_jaune,           # hauteur de la coiffe
    fleche,                  # flèche verticale de l'aileron
    hauteur_gauche_verte,    # saumon (hauteur du bord gauche, robuste)
    hauteur_droite_verte,    # emplanture (hauteur bord droit, robuste)
    envergure,               # largeur totale de l'aileron
    masque_jaune             # masque binaire de la coiffe
    # cleaned,                 # masque nettoyé de tous les éléments
    # aileron_gauche,          # liste des pixels de l'aileron gauche
    # x_emplanture,            # coordonnée X du bord droit de l'aileron (emplanture)
    # x_saumon,                 # coordonnée X du bord gauche de l'aileron (saumon)
    # y_gauche_min,
    # y_gauche_max,
    # y_droite_min,
    # y_droite_max,
    # masque_rouge,
    # masque_vert
    ) = analyse_image(image)

    Donnees = {}

    Donnees["Long_ogive"] = int(hauteur_jaune) # Hauteur de l'ogive
    Donnees["D_og"] = int(epaisseur_moyenne_rouge) # Diamètre à la base de l'ogive
    Donnees["D_ref"] = int(Donnees["D_og"]) # Égal au diamètre de l'ogive
    Donnees["Nb_trans"] = int(0) # Nombre de transitions

    # Ailerons du bas
    Donnees["m_ail"] = int(hauteur_droite_verte) # Emplanture m
    Donnees["n_ail"] = int(hauteur_gauche_verte) # Saumon n
    Donnees["p_ail"] = int(fleche) # Flèche p
    Donnees["E_ail"] = int(envergure) # Envergure E
    Donnees["X_ail"] = int(hauteur_rouge + hauteur_jaune) # Position du bas
    Donnees["D_ail"] = int(epaisseur_moyenne_rouge) # Diamètre aux ailerons

    # Ailerons du haut (canard)
    Donnees["m_can"] = int(60) # Emplanture m
    Donnees["n_can"] = int(25) # Saumon n
    Donnees["p_can"] = int(20) # Flèche p
    Donnees["E_can"] = int(45) # Envergure E
    Donnees["X_can"] = int(500) # Position du bas
    Donnees["D_can"] = int(54) # Diamètre aux ailerons

    Donnees["Long_tot"] = int(hauteur_rouge + hauteur_jaune) # Longueur totale

    # Transition 1
    Donnees["l_j"] = int(30) # Longueur de la transition
    Donnees["D1j"] = Donnees["D_og"] # Diamètre en haut de la transition
    Donnees["D2j"] = int(50) # Diamètre en bas de la transition
    Donnees["X_j"] = int(300) # Position depuis le sommet de l'ogive

    # Transition 2
    Donnees["l_r"] = int(27) # Longueur de la transition
    Donnees["D1r"] = int(Donnees["D2j"]) # Diamètre en haut de la transition
    Donnees["D2r"] = int(54) # Diamètre en bas de la transition
    Donnees["X_r"] = int(360) # Position depuis le sommet de l'ogive

    # Transition 3
    Donnees["l_s"] = int(10) # Longueur de la transition
    Donnees["D1s"] = int(Donnees["D2r"]) # Diamètre en haut de la transition
    Donnees["D2s"] = int(10) # Diamètre en bas de la transition
    Donnees["X_s"] = int(10) # Position depuis le sommet de l'ogive

    Donnees["XpropuRef"] = Donnees["Long_tot"]

    Donnees["Forme_ogive"] = noseConeShape.noseConeType(masque_jaune)

    return Donnees