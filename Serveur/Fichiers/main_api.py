from fastapi import FastAPI, File, UploadFile
from analyse_fusee import analyse_image
from PIL import Image
import io

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lire l'image envoyée par l'app
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes))

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

    Donnes = {}

    Donnes["Long_ogive"] = int(hauteur_jaune) # Hauteur de l'ogive
    Donnes["D_og"] = int(epaisseur_moyenne_rouge) # Diamètre à la base de l'ogive
    Donnes["D_ref"] = int(Donnes["D_og"]) # Égal au diamètre de l'ogive
    Donnes["Nb_trans"] = int(0) # Nombre de transitions

    # Ailerons du bas
    Donnes["m_ail"] = int(hauteur_droite_verte) # Emplanture m
    Donnes["n_ail"] = int(hauteur_gauche_verte) # Saumon n
    Donnes["p_ail"] = int(fleche) # Flèche p
    Donnes["E_ail"] = int(envergure) # Envergure E
    Donnes["X_ail"] = int(hauteur_rouge + hauteur_jaune) # Position du bas
    Donnes["D_ail"] = int(epaisseur_moyenne_rouge) # Diamètre aux ailerons

    # Ailerons du haut (canard)
    Donnes["m_can"] = int(60) # Emplanture m
    Donnes["n_can"] = int(25) # Saumon n
    Donnes["p_can"] = int(20) # Flèche p
    Donnes["E_can"] = int(45) # Envergure E
    Donnes["X_can"] = int(500) # Position du bas
    Donnes["D_can"] = int(54) # Diamètre aux ailerons

    Donnes["Long_tot"] = int(hauteur_rouge + hauteur_jaune) # Longueur totale

    # Transition 1
    Donnes["l_j"] = int(30) # Longueur de la transition
    Donnes["D1j"] = Donnes["D_og"] # Diamètre en haut de la transition
    Donnes["D2j"] = int(50) # Diamètre en bas de la transition
    Donnes["X_j"] = int(300) # Position depuis le sommet de l'ogive

    # Transition 2
    Donnes["l_r"] = int(27) # Longueur de la transition
    Donnes["D1r"] = int(Donnes["D2j"]) # Diamètre en haut de la transition
    Donnes["D2r"] = int(54) # Diamètre en bas de la transition
    Donnes["X_r"] = int(360) # Position depuis le sommet de l'ogive

    # Transition 3
    Donnes["l_s"] = int(10) # Longueur de la transition
    Donnes["D1s"] = int(Donnes["D2r"]) # Diamètre en haut de la transition
    Donnes["D2s"] = int(10) # Diamètre en bas de la transition
    Donnes["X_s"] = int(10) # Position depuis le sommet de l'ogive

    Donnes["XpropuRef"] = Donnes["Long_tot"]

    img_ogive = masque_jaune # Ogive segmentée pour Pablo

    img_ogive_list = img_ogive.tolist()

    return {
        "donnees": Donnes,
        "img_ogive": img_ogive_list,
        "img_ogive_shape": img_ogive.shape
    }