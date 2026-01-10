from scipy.ndimage import label, center_of_mass, binary_fill_holes
import segmentation_models_pytorch as smp
import numpy as np
import torch
import cv2

#%%

# ------------------------------------------------------------
# DFS pour les composantes connexes
# En gros cest un algorithme qui permet permet de détectés les 
# groupe de pixel. Il prend un pixel blanc, cherche autour de lui
# si il y a un autre pixel blanc, si cest le cas il le ratache au
# groupe du premier et notifie qu'il a ete identifie, pour eviter
# de retomber dessus. Au final ces fonction permet de generer des
# listes contennat les coordonnees de pixels colles entre eux
# ------------------------------------------------------------
def dfs(mask, y, x, visited):
    stack = [(y, x)]
    comp = []
    visited[y, x] = True
    H, W = mask.shape
    while stack:
        cy, cx = stack.pop()
        comp.append((cy, cx))
        for ny, nx in [(cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)]:
            if 0 <= ny < H and 0 <= nx < W:
                if not visited[ny,nx] and mask[ny,nx]:
                    visited[ny,nx] = True
                    stack.append((ny,nx))
    return comp

def connected_components(mask):
    H, W = mask.shape
    visited = np.zeros((H,W), dtype=bool)
    components = []
    for y in range(H):
        for x in range(W):
            if mask[y,x]==1 and not visited[y,x]:
                components.append(dfs(mask,y,x,visited))
    return components

# ------------------------------------------------------------
# Chargement image
# ------------------------------------------------------------
def load_image(image, target_size=(512,512), device="cpu"):
    img = image.convert("RGB")
    original_size = img.size  # (width, height) - GARDER LA TAILLE ORIGINALE
    img_resized = img.resize(target_size)
    img_np = np.array(img_resized, dtype=np.float32)/255.0
    img_tensor = torch.tensor(img_np).permute(2,0,1).unsqueeze(0)
    return img_tensor.to(device), img_np, original_size  # RETOURNER LA TAILLE ORIGINALE


# ------------------------------------------------------------
# Cette fonction permet d'appliquer un facteur de correction
# aux mesures en fonction de la distance entre l'appareil photo
# et le fond Zmires, et la distance entre la fusée et le fond
# ------------------------------------------------------------
def facteur_correction(Z_mires, d_fusee):
    return Z_mires / (Z_mires - d_fusee)


def load_image2(image, target_size=(512,512), device="cpu"):
    """
    Charge une image PIL, redimensionne pour le réseau, normalise et retourne le tenseur.
    
    Arguments :
    - image : objet PIL.Image
    - target_size : tuple (largeur, hauteur) pour redimensionner l'image
    - device : "cpu" ou "cuda"
    
    Retour :
    - img_tensor : tenseur normalisé prêt pour le réseau (1,3,H,W)
    - img_np : image redimensionnée en numpy (H,W,3)
    - original_size : taille originale de l'image (width, height)
    """
    # Garder la taille originale
    original_size = image.size  # (width, height)
    
    # Redimensionner pour le réseau
    img_resized = image.resize(target_size)
    
    # Convertir en numpy float32 et normaliser entre 0 et 1
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Convertir en tenseur (C,H,W)
    img_tensor = torch.tensor(img_np).permute(2,0,1).unsqueeze(0)
    
    # Normalisation ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    img_tensor = (img_tensor - mean) / std
    
    # Envoyer sur le bon device
    img_tensor = img_tensor.to(device)
    
    return img_tensor, img_np, original_size


# ------------------------------------------------------------
# Fonction principale. Ici on va utiliser le réseaux de neuronnes
# Sur notre image afin d'en extraire les dimensions (pour l'instant)
# en pixels.
# ------------------------------------------------------------
def analyse_image(path_image):

    # ─────────────────────────────────────────────────────────────────────────────
    # Chargement du modèle de détection des mires
    # ─────────────────────────────────────────────────────────────────────────────

    # Ici on va venir charger le réseau de neuronnes préentraine pour la détection des mires
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=4
    )

    model.load_state_dict(torch.load("modele_mires.pth", map_location=device))# On charge les poids du modèle entraîné
    #checkpoint = torch.load("modele_mires.pth", map_location=device)
    #model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Prédiction du masque des mires
    # ─────────────────────────────────────────────────────────────────────────────

    img_tensor, img_np, original_size = load_image(path_image, device=device)# Charge et redimensionne l'image pour le réseau et on garde les dimensions originales

    with torch.no_grad():
        pred = model(img_tensor)
        pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    
    # On redimensionne le masque aux dimensions originales de l'image
    pred_mask_original = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Extraction et nettoyage des mires détectées
    # ─────────────────────────────────────────────────────────────────────────────
    mask_mires_brut = (pred_mask_original == 1).astype(int)
    labeled_array, num_features = label(mask_mires_brut)
    #print(f"Nombre de groupes détectés : {num_features}")
    
    # ═════════════════════════════════════════════════════════════════════════════
    # AU MOINS 4 MIRES DÉTECTÉES
    # ═════════════════════════════════════════════════════════════════════════════

    if num_features >= 4:
            
        # ─────────────────────────────────────────────────────────────────────────
        # Sélection des 4 plus gros groupes (les vraies mires)
        # ─────────────────────────────────────────────────────────────────────────

        # Pour chaque groupe, on calcule sa distance minimale aux bords gauche et droit
        edge_distances = []
        width = labeled_array.shape[1]  # Largeur de l'image

        for i in range(1, num_features + 1):
            # Trouve les coordonnées de tous les pixels du groupe i
            coords = np.argwhere(labeled_array == i)
            # coords[:, 1] contient les coordonnées x (colonnes)
            x_coords = coords[:, 1]
            
            # Distance minimale au bord gauche (x=0) ou droit (x=width-1)
            min_dist_left = np.min(x_coords)
            min_dist_right = width - 1 - np.max(x_coords)
            min_edge_distance = min(min_dist_left, min_dist_right)
            
            edge_distances.append(min_edge_distance)

        # On trie les groupes par distance croissante aux bords (les plus proches en premier)
        sorted_labels = [i + 1 for i in np.argsort(edge_distances)]
        top_4_labels = sorted_labels[:4]  # On garde les 4 plus proches des bords

        
        # On crée maitenant un masque nettoyé avec seulement les 4 plus gros groupes
        mask_mires_clean = np.zeros_like(mask_mires_brut)
        for lbl in top_4_labels:
            mask_mires_clean[labeled_array == lbl] = 1
        
        # On Re-labellise le masque nettoyé (labels 1, 2, 3, 4) pour reconnaitre précisément chaque mires
        labeled_array_clean, _ = label(mask_mires_clean)

        # Trouver les colonnes où il y a des pixels à 1
        cols_with_mires = np.where(mask_mires_clean.any(axis=0))[0]

        if len(cols_with_mires) > 0:
            left_bound = cols_with_mires[0]
            right_bound = cols_with_mires[-1]
        else:
            # Fallback si aucune mire n'est détectée
            left_bound = 0
            right_bound = mask_mires_clean.shape[1] - 1

        mask_mires_clean = mask_mires_clean[:, left_bound:right_bound+1]

        # ─────────────────────────────────────────────────────────────────────────
        # Calcul des centres de masse des 4 mires
        # ─────────────────────────────────────────────────────────────────────────

        centers = []
        for lbl in range(1, 5):  # Labels 1, 2, 3, 4
            # On isole le groupe de pixels de cette mire
            group_mask = (labeled_array_clean == lbl).astype(int)
            # On calcule son centre de masse (coordonnées moyennes pondérées)
            center = center_of_mass(group_mask)
            centers.append(center)
            #print(f"Centre mire {lbl}: ({center[0]:.1f}, {center[1]:.1f})")
        
        # ─────────────────────────────────────────────────────────────────────────
        # Identification des 4 coins du rectangle de référence
        # ─────────────────────────────────────────────────────────────────────────
        
        # On trie les centres par coordonnée Y (du haut vers le bas)
        centers_sorted_y = sorted(centers, key=lambda c: c[0])
        
        # On sépare les 2 mires du haut et les 2 du bas
        top_2 = sorted(centers_sorted_y[:2], key=lambda c: c[1])     # Trié par X
        bottom_2 = sorted(centers_sorted_y[2:], key=lambda c: c[1])  # Trié par X
        
        # Assigne chaque coin
        mire_haut_gauche = top_2[0]      # Coin supérieur gauche
        mire_haut_droite = top_2[1]      # Coin supérieur droit
        mire_bas_gauche = bottom_2[0]    # Coin inférieur gauche
        mire_bas_droite = bottom_2[1]    # Coin inférieur droit
        
        # print("\n=== POSITION DES MIRES ===")
        # print(f"Haut-Gauche  : ({mire_haut_gauche[0]:.1f}, {mire_haut_gauche[1]:.1f})")
        # print(f"Haut-Droite  : ({mire_haut_droite[0]:.1f}, {mire_haut_droite[1]:.1f})")
        # print(f"Bas-Gauche   : ({mire_bas_gauche[0]:.1f}, {mire_bas_gauche[1]:.1f})")
        # print(f"Bas-Droite   : ({mire_bas_droite[0]:.1f}, {mire_bas_droite[1]:.1f})")

    else:
        raise ValueError(f"Calibration impossible : seulement {num_features} mire(s) détectée(s). Il en faut au moins 4.")

    # ════════════════════════════════════════════════════════════════════════════════
    # ███████╗██╗   ██╗███████╗███████╗███████╗
    # ██╔════╝██║   ██║██╔════╝██╔════╝██╔════╝
    # █████╗  ██║   ██║███████╗█████╗  █████╗  
    # ██╔══╝  ██║   ██║╚════██║██╔══╝  ██╔══╝  
    # ██║     ╚██████╔╝███████║███████╗███████╗
    # ╚═╝      ╚═════╝ ╚══════╝╚══════╝╚══════╝
    #                                           
    # ════════════════════════════════════════════════════════════════════════════════
    #                           ANALYSE DE LA FUSÉE
    # ════════════════════════════════════════════════════════════════════════════════
        
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.Unet( # Ici on va venir créer le réseau de neuronnes, deja entraines pour l'analyse de la fusee
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4
    )
    model.load_state_dict(torch.load("modele_fusee_V3.pth", map_location=device)) # on charge les poids
    #checkpoint = torch.load("modele_fusee_V3.pth", map_location=device)
    #model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    path_image = path_image.crop((left_bound, 0, right_bound + 1, path_image.height))
    img_tensor, img_np, original_size = load_image2(path_image, device=device)  #Maintenant on va redimensionner l'image que l'on souhaite analyser pour le réseau
    #, mais surtout garder les dimensions de l'image d'origine

    # # Prédiction
    # with torch.no_grad():
    #     pred = model(img_tensor)
    #     pred_mask = torch.argmax(pred.squeeze(), dim=0).cpu().numpy()# Ici on recupere le masque genere par le reseau, donc le fuselage cone et aielrons

    # pred_mask_original = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST) # On redimensionne ici le masque du reseau au vraies dimensions de l'iamge

    pred = model(img_tensor)  # (1, C, H, W)

    # Resize des logits (float)
    pred_resized = torch.nn.functional.interpolate(
        pred,
        size=(original_size[1], original_size[0]),  # (H, W)
        mode="bilinear",
        align_corners=False
    )

    # Argmax APRÈS resize
    pred_mask_original = torch.argmax(
        pred_resized.squeeze(0),
        dim=0
    ).cpu().numpy()

    cleaned = np.zeros_like(pred_mask_original) # ici on prend une image vide
    comps = {1:[],2:[],3:[]}# Dictionnaire pour stocker les groupes de pixels par classe (1=fuselage, 2=coiffe, 3=ailerons)
    
    for cls in [1,2,3]:# Pour chaque classe détectée par le réseau
        cls_mask = (pred_mask_original==cls).astype(np.uint8) # Ici on extrait un masque binaire : 1 où le pixel = classe actuelle, 0 ailleurs
        
        # On trouve tous les groupes de pixels connectés pour cette classe
        comps[cls] = connected_components(cls_mask)
        
    # Pour le masque rouge, on ne garde que le groupe possedant le plus de pixels (logiquement cest le fuselage qui sera gardé)
    if comps[1]:
        largest = max(comps[1], key=len)
        for y,x in largest: cleaned[y,x]=1
        comps[1] = [largest]

    # Pour le masque jaune, meme reflexion on ne garde que le groupe possedant le plus de pixels (logiquement cest le coiffe qui sera gardé)
    if comps[2]:
        largest = max(comps[2], key=len)
        for y,x in largest: cleaned[y,x]=2
        comps[2] = [largest]

    # Pour le masque vert cest différent, il se peut que l'on voit les 4 ailerons sur l'image, donc on supprimer les groupes a partir du 5ieme si ils existent
    comps[3].sort(key=len, reverse=True)
    comps[3] = [c for c in comps[3] if len(c)>=4][:4]
    for comp in comps[3]:
        for y,x in comp: cleaned[y,x]=3

    # ════════════════════════════════════════════════════════════════════════════════
    # ███████╗██╗   ██╗███████╗███████╗██╗      █████╗  ██████╗ ███████╗
    # ██╔════╝██║   ██║██╔════╝██╔════╝██║     ██╔══██╗██╔════╝ ██╔════╝
    # █████╗  ██║   ██║███████╗█████╗  ██║     ███████║██║  ███╗█████╗  
    # ██╔══╝  ██║   ██║╚════██║██╔══╝  ██║     ██╔══██║██║   ██║██╔══╝  
    # ██║     ╚██████╔╝███████║███████╗███████╗██║  ██║╚██████╔╝███████╗
    # ╚═╝      ╚═════╝ ╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
    #                                           
    # ════════════════════════════════════════════════════════════════════════════════
    #                          MESURES FUSELAGE (ROUGE)
    # ════════════════════════════════════════════════════════════════════════════════
        
    
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HAUTEUR FUSELAGE
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Ici on va venir calculer en longueur de pixels la longueur du fuselage, et son diamètre moyen
    if comps[1]: 
        red = comps[1][0] # groupe le plus gros de pixels rouge
        ys = [p[0] for p in red]
        xs = [p[1] for p in red]
        hauteur_rouge = max(ys)-min(ys) # On prend la hauteur maximale des pixels


        # ═══════════════════════════════════════════════════════════════════════════
        # DIAMETRE FUSELAGE
        # ═══════════════════════════════════════════════════════════════════════════
        
        temp = np.zeros_like(cleaned) # Ici on va venir calculer la moyenne de la largeur du fuselage à chaque ligne
        for y,x in red: temp[y,x]=1
        epaisseurs = []
        for y in range(temp.shape[0]):
            xs_l = np.where(temp[y]==1)[0]
            if len(xs_l)>1:
                epaisseurs.append(xs_l.max()-xs_l.min())
        epaisseur_moyenne_rouge = np.mean(epaisseurs) if epaisseurs else 0
        
        
    else:# Si aucun groupe de pixels rouge n'a été détecté, on renvoie des valeurs nulles
        hauteur_rouge = epaisseur_moyenne_rouge = 0
    
    # ════════════════════════════════════════════════════════════════════════════════
    #  ██████╗ ██████╗ ██╗███████╗███████╗███████╗
    # ██╔════╝██╔═══██╗██║██╔════╝██╔════╝██╔════╝
    # ██║     ██║   ██║██║█████╗  █████╗  █████╗  
    # ██║     ██║   ██║██║██╔══╝  ██╔══╝  ██╔══╝  
    # ╚██████╗╚██████╔╝██║██║     ██║     ███████╗
    #  ╚═════╝ ╚═════╝ ╚═╝╚═╝     ╚═╝     ╚══════╝
    #                                           
    # ════════════════════════════════════════════════════════════════════════════════
    #                           MESURES COIFFE (JAUNE)
    # ════════════════════════════════════════════════════════════════════════════════
        
    # ═══════════════════════════════════════════════════════════════════════════
    # HAUTEUR COIFFE
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Ici on va venir calculer en longueur de pixels la longueur de la coiffe
    if comps[2]: 
        yellow = comps[2][0] # groupe le plus gros de pixels jaune
        ys = [p[0] for p in yellow]
        hauteur_jaune = max(ys)-min(ys) # On prend la hauteur maximale des pixels
    else:# Si aucun groupe de pixels jaune n'a été détecté, on renvoie des valeurs nulles
        hauteur_jaune = 0

    # ════════════════════════════════════════════════════════════════════════════════
    #  █████╗ ██╗██╗     ███████╗██████╗  ██████╗ ███╗   ██╗
    # ██╔══██╗██║██║     ██╔════╝██╔══██╗██╔═══██╗████╗  ██║
    # ███████║██║██║     █████╗  ██████╔╝██║   ██║██╔██╗ ██║
    # ██╔══██║██║██║     ██╔══╝  ██╔══██╗██║   ██║██║╚██╗██║
    # ██║  ██║██║███████╗███████╗██║  ██║╚██████╔╝██║ ╚████║
    # ╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
    #                                           
    # ════════════════════════════════════════════════════════════════════════════════
    #                      MESURES AILERON GAUCHE (VERT)
    # ════════════════════════════════════════════════════════════════════════════════

    if comps[3]:
        
        # Ici on trie les ailerons de gauche à droite selon leur position X moyenne
        ailerons_tries = sorted(comps[3], key=lambda c: np.mean([p[1] for p in c]))
        aileron_gauche = ailerons_tries[0] # On prend l'aileron le plus à gauche
        
        # On extrait toutes les coordonnées Y et X de l'aileron
        ys = np.array([p[0] for p in aileron_gauche])
        xs = np.array([p[1] for p in aileron_gauche])
        
        # ═══════════════════════════════════════════════════════════════════════════
        # ENVERGURE (largeur totale de l'aileron)
        # ═══════════════════════════════════════════════════════════════════════════
    

        x_min, x_max = xs.min(), xs.max()
        envergure = x_max - x_min # Largeur totale de l'aileron

        # ═══════════════════════════════════════════════════════════════════════════
        # SAUMON (corde au bord d'attaque, côté gauche)
        # ═══════════════════════════════════════════════════════════════════════════
    
        # On définit une zone de recherche : les 10% les plus à gauche, 
        # pourquoi parce que le réseau de neuronnes nest pas précis donc parfait 
        # les pixels les plus a gauche ne sont seulement une colonne de 2 pixels
        x_threshold = x_min + int(0.10 * envergure)
        
        hauteurs_colonnes = [] # Liste pour stocker les hauteurs trouvées dans cette zone
        
        for x in range(x_min, x_threshold + 1):
            ys_col = ys[xs == x]
            if len(ys_col) > 1: # Si au moins 2 pixels, on calcule la hauteur de la colonne
                hauteur_col = ys_col.max() - ys_col.min()
                hauteurs_colonnes.append((hauteur_col, x, ys_col.min(), ys_col.max()))
        
        if hauteurs_colonnes:     # Si on a trouvé des colonnes valides
            # On prend la colonne avec la plus grande hauteur 
            hauteur_gauche_verte, x_saumon, y_gauche_min, y_gauche_max = max( hauteurs_colonnes, key=lambda t: t[0] )
        
        else:
            ys_g = ys[xs == x_min]
            y_gauche_min, y_gauche_max = ys_g.min(), ys_g.max()
            hauteur_gauche_verte = y_gauche_max - y_gauche_min
            x_saumon = x_min

        # ═══════════════════════════════════════════════════════════════════════════
        # EMPLANTURE (corde au bord de fuite, côté droit)
        # ═══════════════════════════════════════════════════════════════════════════
    
        # Meme raisonnement que pour le saumon, mais cette fois ci pour la droite, 
        # et cette fois avec 30% (parfois le masque de l'aileron empiete sur le fuselage)
        x_threshold = x_max - int(0.30 * envergure)
        
        hauteurs_colonnes = [] # Liste pour stocker les hauteurs trouvées dans cette zone
        
        for x in range(x_threshold, x_max + 1):
            ys_col = ys[xs == x]
            if len(ys_col) > 1: # Si au moins 2 pixels, on calcule la hauteur de la colonne
                hauteur_col = ys_col.max() - ys_col.min()
                hauteurs_colonnes.append((hauteur_col, x, ys_col.min(), ys_col.max()))
        
        if hauteurs_colonnes: # Si on a trouvé des colonnes valides
            # On prend la colonne avec la plus grande hauteur 

            hauteur_droite_verte, x_emplanture, y_droite_min, y_droite_max = max(hauteurs_colonnes, key=lambda t: t[0] )
        else:
            ys_d = ys[xs == x_max]
            y_droite_min, y_droite_max = ys_d.min(), ys_d.max()
            hauteur_droite_verte = y_droite_max - y_droite_min
            x_emplanture = x_max


        # ═══════════════════════════════════════════════════════════════════════════
        # FLÈCHE (décalage vertical entre le saumon et l'emplanture)
        # ═══════════════════════════════════════════════════════════════════════════
    
        fleche = abs(y_gauche_min - y_droite_min)  # Flèche = différence verticale entre le bord avant (gauche) et arrière (droit)
    
    else:
        fleche= hauteur_gauche_verte=hauteur_droite_verte=envergure=0 # Si aucun groupe de pixels verts n'a été détecté, on renvoie des valeurs nulles

    # ════════════════════════════════════════════════════════════════════════════════
    # ███╗   ███╗███████╗███████╗██╗   ██╗██████╗ ███████╗███████╗
    # ████╗ ████║██╔════╝██╔════╝██║   ██║██╔══██╗██╔════╝██╔════╝
    # ██╔████╔██║█████╗  ███████╗██║   ██║██████╔╝█████╗  ███████╗
    # ██║╚██╔╝██║██╔══╝  ╚════██║██║   ██║██╔══██╗██╔══╝  ╚════██║
    # ██║ ╚═╝ ██║███████╗███████║╚██████╔╝██║  ██║███████╗███████║
    # ╚═╝     ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝
    #                                           
    # ════════════════════════════════════════════════════════════════════════════════
    #                      CONVERSION DES LONGUEURS EN MÈTRES
    # ════════════════════════════════════════════════════════════════════════════════
        
    masque_jaune = (cleaned==2).astype(np.uint8) # pour l'envoyer au code de pablo
    masque_rouge = (cleaned==1).astype(np.uint8) # pour l'envoyer au code de pablo
    masque_vert = (pred_mask_original==3).astype(np.uint8) # pour l'envoyer au code de pablo
   
    # ─────────────────────────────────────────────────────────────────────────
    # Calcul des échelles de conversion pixels à mètres
    # ─────────────────────────────────────────────────────────────────────────
    
    # Hypothèse : les mires sont espacées de 1 mètre réel
            
    DISTANCE_VERTICALE_MIRES = 0.9 # Distance entre mires haut et bas
    DISTANCE_HORIZONTALE_MIRES = 0.8 # Distance entre mires gauche et droite

    echelle_verticale_gauche = abs(mire_bas_gauche[0] - mire_haut_gauche[0]) / DISTANCE_VERTICALE_MIRES
    echelle_verticale_droite = abs(mire_bas_droite[0] - mire_haut_droite[0]) / DISTANCE_VERTICALE_MIRES
    echelle_horizontale_haut = abs(mire_haut_droite[1] - mire_haut_gauche[1]) / DISTANCE_HORIZONTALE_MIRES
    echelle_horizontale_bas = abs(mire_bas_droite[1] - mire_bas_gauche[1]) / DISTANCE_HORIZONTALE_MIRES 
    
    # print("\n=== ÉCHELLES CALCULÉES ===")
    # print(f"Verticale gauche  : {echelle_verticale_gauche:.2f} px/m")
    # print(f"Verticale droite  : {echelle_verticale_droite:.2f} px/m")
    # print(f"Horizontale haut  : {echelle_horizontale_haut:.2f} px/m")
    # print(f"Horizontale bas   : {echelle_horizontale_bas:.2f} px/m")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Détection de la distorsion perspective
    # ─────────────────────────────────────────────────────────────────────────
    
    # plus les échelles (verticale et horizontale) diffèrent beaucoup, plus l'image a une forte perspective, 
    # et donc plus les mesures peuvent être faussées
    # distorsion_verticale = abs(echelle_verticale_gauche - echelle_verticale_droite)
    # distorsion_horizontale = abs(echelle_horizontale_haut - echelle_horizontale_bas)
    
    # if distorsion_verticale > 10:
    #     print(f" Distorsion perspective verticale détectée : {distorsion_verticale:.2f} px")
    # if distorsion_horizontale > 10:
    #     print(f" Distorsion perspective horizontale détectée : {distorsion_horizontale:.2f} px")
        
    # ─────────────────────────────────────────────────────────────────────────
    # Fonction d'interpolation bilinéaire de l'échelle
    # ─────────────────────────────────────────────────────────────────────────
                
    def calculer_echelle_verticale_locale(y, x):
        """
        Calcule l'échelle VERTICALE locale (pour hauteurs).
        Retourne px/m pour les distances verticales.
        """
        # Calcul des ratios
        hauteur_rect = (mire_bas_gauche[0] + mire_bas_droite[0]) / 2 - \
                        (mire_haut_gauche[0] + mire_haut_droite[0]) / 2
        largeur_rect = (mire_haut_droite[1] + mire_bas_droite[1]) / 2 - \
                        (mire_haut_gauche[1] + mire_bas_gauche[1]) / 2
        
        y_min = (mire_haut_gauche[0] + mire_haut_droite[0]) / 2
        x_min = (mire_haut_gauche[1] + mire_bas_gauche[1]) / 2
        
        ratio_y = max(0, min(1, (y - y_min) / hauteur_rect if hauteur_rect > 0 else 0.5))
        ratio_x = max(0, min(1, (x - x_min) / largeur_rect if largeur_rect > 0 else 0.5))
        
        # Interpolation des échelles VERTICALES
        echelle_locale = echelle_verticale_gauche * (1 - ratio_x) + \
                            echelle_verticale_droite * ratio_x
        
        # Note : On pourrait aussi interpoler verticalement, mais pour les échelles
        # verticales, la variation est surtout horizontale (perspective gauche-droite)
        
        return echelle_locale
    
    
    def calculer_echelle_horizontale_locale(y, x):
        """
        Calcule l'échelle HORIZONTALE locale (pour largeurs, envergures).
        Retourne px/m pour les distances horizontales.
        """
        # Calcul des ratios
        hauteur_rect = (mire_bas_gauche[0] + mire_bas_droite[0]) / 2 - \
                        (mire_haut_gauche[0] + mire_haut_droite[0]) / 2
        largeur_rect = (mire_haut_droite[1] + mire_bas_droite[1]) / 2 - \
                        (mire_haut_gauche[1] + mire_bas_gauche[1]) / 2
        
        y_min = (mire_haut_gauche[0] + mire_haut_droite[0]) / 2
        x_min = (mire_haut_gauche[1] + mire_bas_gauche[1]) / 2
        
        ratio_y = max(0, min(1, (y - y_min) / hauteur_rect if hauteur_rect > 0 else 0.5))
        ratio_x = max(0, min(1, (x - x_min) / largeur_rect if largeur_rect > 0 else 0.5))
        
        # Interpolation des échelles HORIZONTALES
        echelle_locale = echelle_horizontale_haut * (1 - ratio_y) + \
                            echelle_horizontale_bas * ratio_y
        
        # Note : On pourrait aussi interpoler horizontalement, mais pour les échelles
        # horizontales, la variation est surtout verticale (perspective haut-bas)
        
        return echelle_locale
    
    # ─────────────────────────────────────────────────────────────────────────
    # Calcul des positions moyennes des éléments de la fusée
    # ─────────────────────────────────────────────────────────────────────────

    # Position moyenne du fuselage (pour adapter l'échelle locale)
    if comps[1]:  # Fuselage rouge
        y_moyen_fuselage = np.mean([p[0] for p in comps[1][0]])
        x_moyen_fuselage = np.mean([p[1] for p in comps[1][0]])
    else:
        y_moyen_fuselage = original_size[1] / 2
        x_moyen_fuselage = original_size[0] / 2
    
    
    # Position moyenne de la coiffe
    if comps[2]:
        y_moyen_coiffe = np.mean([p[0] for p in comps[2][0]])
        x_moyen_coiffe = np.mean([p[1] for p in comps[2][0]])
    else:
        y_moyen_coiffe = y_moyen_fuselage  # Fallback
        x_moyen_coiffe = x_moyen_fuselage
        
        
    # Position moyenne de l'aileron
    if comps[3]:  # Aileron vert
        y_moyen_aileron = np.mean([p[0] for p in comps[3][0]])
        x_moyen_aileron = np.mean([p[1] for p in comps[3][0]])
    else:
        y_moyen_aileron = y_moyen_fuselage
        x_moyen_aileron = x_moyen_fuselage
    
    # ─────────────────────────────────────────────────────────────────────────
    # Conversion des longueurs en mètres
    # ─────────────────────────────────────────────────────────────────────────

    # Calcul du facteur de correction du au fait que la fusée et le fond 
    # ne sont pas dans le meme plan
    facteur = facteur_correction(1.0, 0.20)  # Hypothèse appareil photo a 1.20 m du fond, et fusée séparé de 13 cm du fond

    # FUSELAGE
    echelle_vert_fuselage = calculer_echelle_verticale_locale(y_moyen_fuselage, x_moyen_fuselage)
    echelle_horiz_fuselage = calculer_echelle_horizontale_locale(y_moyen_fuselage, x_moyen_fuselage)
    
    hauteur_rouge_m = (hauteur_rouge / echelle_vert_fuselage) * 1000 / facteur  # Hauteur = vertical
    epaisseur_moyenne_rouge_m = (epaisseur_moyenne_rouge / echelle_horiz_fuselage) * 1000 / facteur  # Diamètre = horizontal

    # COIFFE
    echelle_vert_coiffe = calculer_echelle_verticale_locale(y_moyen_coiffe, x_moyen_coiffe)
    hauteur_jaune_m = (hauteur_jaune / echelle_vert_coiffe) * 1000 / facteur  # Hauteur = vertical
    
    # AILERON
    echelle_vert_saumon = calculer_echelle_verticale_locale((y_gauche_min + y_gauche_max)/2, x_saumon)
    echelle_vert_emplanture = calculer_echelle_verticale_locale((y_droite_min + y_droite_max)/2, x_emplanture)
    echelle_horiz_aileron = calculer_echelle_horizontale_locale(y_moyen_aileron, x_moyen_aileron)
    
    hauteur_gauche_verte_m = (hauteur_gauche_verte / echelle_vert_saumon) * 1000 / facteur  # Hauteur = vertical
    hauteur_droite_verte_m = (hauteur_droite_verte / echelle_vert_emplanture) * 1000 / facteur  # Hauteur = vertical
    envergure_m = (envergure / echelle_horiz_aileron) * 1000 / facteur  # Envergure = horizontal
    
    # Flèche = vertical, moyenne des deux échelles
    echelle_fleche = (echelle_vert_saumon + echelle_vert_emplanture) / 2
    fleche_m = (fleche / echelle_fleche) * 1000 / facteur

        
    return (
        hauteur_rouge_m,           # hauteur du fuselage
        epaisseur_moyenne_rouge_m, # épaisseur moyenne du fuselage
        hauteur_jaune_m,           # hauteur de la coiffe
        fleche_m,                  # flèche verticale de l'aileron
        hauteur_gauche_verte_m,    # saumon (hauteur du bord gauche, robuste)
        hauteur_droite_verte_m,    # emplanture (hauteur bord droit, robuste)
        envergure_m,               # largeur totale de l'aileron
        masque_jaune,              # masque binaire de la coiffe
        # cleaned,                 # masque nettoyé de tous les éléments
        # aileron_gauche,          # liste des pixels de l'aileron gauche
        # x_emplanture,            # coordonnée X du bord droit de l'aileron (emplanture)
        # x_saumon,                # coordonnée X du bord gauche de l'aileron (saumon)
        # y_gauche_min,
        # y_gauche_max,
        # y_droite_min,
        # y_droite_max,
        # masque_rouge,
        # masque_vert
        )