import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter

def intersection_lignes(a_h, b_h, a_v, b_v):
    """
    Calcule l'intersection entre :
    - Une droite type y = ax + b (horizontale/oblique) : params (a_h, b_h)
    - Une droite type x = ay + b (verticale/oblique)   : params (a_v, b_v)
    """
    # Math: x = a_v * (a_h * x + b_h) + b_v
    # x = a_v*a_h*x + a_v*b_h + b_v
    # x * (1 - a_v*a_h) = a_v*b_h + b_v
    
    div = (1 - a_v * a_h)
    if div == 0: 
        return 0, 0 # Droites parallèles (cas très rare ici)
        
    x = (a_v * b_h + b_v) / div
    y = a_h * x + b_h
    return x, y

def lisser_contour(y, window=11, poly=2):
    """
    Lissage 1D d’un contour sans déformer sa tendance globale.
    """
    if len(y) < window:
        return y
    # window doit être impair
    if window % 2 == 0:
        window += 1
    return savgol_filter(y, window_length=window, polyorder=poly)


def filtrer_ailerons(matrice):
    """
    Garde les deux zones avec le plus de 1 selon les règles:
    - Si > 2 zones: garde les 2 plus grandes
    - Si 2 zones: garde la plus petite (l'aileron du milieu est fusionné)
    - Si <= 1 zone: ne change rien
    
    Args:
        matrice: numpy array binaire (0 et 1)
    
    Returns:
        numpy array filtré avec seulement les zones désirées
    """
    # Labelliser les zones connectées
    labeled_array, num_features = ndimage.label(matrice)
    
    # Si 0 ou 1 zone, on retourne la matrice telle quelle
    if num_features <= 1:
        return matrice.copy()
    
    # Compter la taille de chaque zone
    tailles = []
    for i in range(1, num_features + 1):
        taille = np.sum(labeled_array == i)
        tailles.append((i, taille))
    
    # Trier par taille décroissante
    tailles.sort(key=lambda x: x[1], reverse=True)
    
    # Créer la matrice de sortie
    resultat = np.zeros_like(matrice)
    
    if num_features == 2:
        # Cas spécial: 2 zones -> on garde la plus PETITE
        zone_a_garder = tailles[1][0]  # La plus petite (index 1)
        resultat[labeled_array == zone_a_garder] = 1
    else:
        # Plus de 2 zones -> on garde les 2 plus GRANDES
        for i in range(2):
            zone_a_garder = tailles[i][0]
            resultat[labeled_array == zone_a_garder] = 1
    
    return resultat


def separer_ailerons(matrice):
    """
    Sépare les ailerons en deux matrices distinctes (aileron gauche et aileron droit).
    
    Args:
        matrice: numpy array binaire (0 et 1) avec les ailerons filtrés
    
    Returns:
        tuple: (aileron_gauche, aileron_droit) - deux numpy arrays
    """
    # Labelliser les zones connectées
    labeled_array, num_features = ndimage.label(matrice)
    
    if num_features == 0:
        # Pas d'ailerons
        return matrice.copy(), np.zeros_like(matrice)
    elif num_features == 1:
        # Un seul aileron (on le met à gauche)
        return matrice.copy(), np.zeros_like(matrice)
    
    # Trouver les centres de masse de chaque zone pour déterminer gauche/droite
    centres = []
    for i in range(1, num_features + 1):
        centre = ndimage.center_of_mass(matrice, labeled_array, i)
        centres.append((i, centre))
    
    # Trier par position horizontale (colonne)
    centres.sort(key=lambda x: x[1][1])  # x[1][1] = colonne du centre de masse
    
    # Créer les deux matrices
    aileron_gauche = np.zeros_like(matrice)
    aileron_droit = np.zeros_like(matrice)
    
    # Zone la plus à gauche
    aileron_gauche[labeled_array == centres[0][0]] = 1
    
    # Zone la plus à droite
    if len(centres) >= 2:
        aileron_droit[labeled_array == centres[1][0]] = 1
    
    return np.flip(aileron_gauche, 1), aileron_droit


def regression_lineaire(x, y):
    """
    Calcule la régression linéaire y = a*x + b
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        a = 0
    else:
        a = numerator / denominator
    
    b = y_mean - a * x_mean
    
    return a, b


def calculer_erreur_rapide(params, bord_sup_cols, bord_sup_rows, bord_inf_cols, bord_inf_rows):
    """
    Calcule l'erreur en mesurant la distance absolue (L1) entre les points 
    du contour réel et les droites du polygone.
    """
    a_sup, b_sup, a_inf, b_inf = params
    
    # Prédiction des Y pour les bords haut et bas
    y_pred_sup = a_sup * bord_sup_cols + b_sup
    y_pred_inf = a_inf * bord_inf_cols + b_inf
    
    # Somme des erreurs absolues (plus robuste que le carré pour les contours irréguliers)
    erreur_sup = np.sum(np.abs(bord_sup_rows - y_pred_sup))
    erreur_inf = np.sum(np.abs(bord_inf_rows - y_pred_inf))
    
    return erreur_sup + erreur_inf


def point_dans_polygone(x, y, poly_x, poly_y):
    """
    Vérifie si un point (x, y) est à l'intérieur d'un polygone.
    Utilise l'algorithme du ray casting.
    """
    n = len(poly_x)
    inside = False
    
    p1x, p1y = poly_x[0], poly_y[0]
    for i in range(1, n + 1):
        p2x, p2y = poly_x[i % n], poly_y[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def calculer_erreur_polygone_complete(masque_aileron, coords_x, coords_y):
    """
    Calcule l'erreur complète : nombre de 0 dans le polygone + nombre de 1 hors du polygone
    
    Args:
        masque_aileron: masque binaire de l'aileron
        coords_x, coords_y: coordonnées du polygone
    
    Returns:
        int: nombre total d'erreurs
    """
    # Limiter la zone de test à la bounding box du polygone élargie
    col_min = int(min(coords_x)) - 1
    col_max = int(max(coords_x)) + 2
    row_min = int(min(coords_y)) - 1
    row_max = int(max(coords_y)) + 2
    
    erreur = 0
    
    for r in range(max(0, row_min), min(masque_aileron.shape[0], row_max)):
        for c in range(max(0, col_min), min(masque_aileron.shape[1], col_max)):
            dans_poly = point_dans_polygone(c, r, coords_x, coords_y)
            valeur_pixel = masque_aileron[r, c]
            
            if dans_poly and valeur_pixel == 0:
                # Faux positif : 0 dans le polygone
                erreur += 1
            elif not dans_poly and valeur_pixel == 1:
                # Faux négatif : 1 hors du polygone
                erreur += 1
    
    return erreur


def approximer_aileron_polygone(masque_aileron, seuil_triangle=0.141):
    """
    Approxime un aileron par un trapèze ou un triangle selon un seuil.
    """
    # Trouver les coordonnées de tous les pixels de l'aileron
    rows, cols = np.where(masque_aileron == 1)
    
    if len(rows) == 0:
        return [], []
    
    col_min = cols.min()
    col_max = cols.max()
    
    # Extraire les contours supérieur et inférieur
    bord_sup_cols = []
    bord_sup_rows = []
    bord_inf_cols = []
    bord_inf_rows = []
    
    for col in range(col_min, col_max + 1):
        mask_col = cols == col
        if mask_col.any():
            rows_col = rows[mask_col]
            
            # Bord supérieur
            bord_sup_cols.append(col)
            bord_sup_rows.append(rows_col.max()) 
            
            # Bord inférieur (CORRECTION DU BUG PRÉCÉDENT ICI)
            bord_inf_cols.append(col)      
            bord_inf_rows.append(rows_col.min())
            
    # Convertir en arrays numpy
    bord_sup_cols = np.array(bord_sup_cols)
    bord_sup_rows = np.array(bord_sup_rows)
    bord_inf_cols = np.array(bord_inf_cols)
    bord_inf_rows = np.array(bord_inf_rows)
    
    # Lissage optionnel (si la fonction lisser_contour existe dans ton code)
    # bord_sup_rows = lisser_contour(bord_sup_rows)
    # bord_inf_rows = lisser_contour(bord_inf_rows)
    
    if len(bord_sup_cols) < 2 or len(bord_inf_cols) < 2:
        return [], []

    # Régression linéaire initiale
    a_sup_init, b_sup_init = regression_lineaire(bord_sup_cols, bord_sup_rows)
    a_inf_init, b_inf_init = regression_lineaire(bord_inf_cols, bord_inf_rows)
    
    # Optimisation
    params_init = [a_sup_init, b_sup_init, a_inf_init, b_inf_init]
    
    result = minimize(
        calculer_erreur_rapide,
        params_init,
        args=(bord_sup_cols, bord_sup_rows, bord_inf_cols, bord_inf_rows),
        method='Nelder-Mead', 
        options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-3}
    )
    
    a_sup, b_sup, a_inf, b_inf = result.x
    
    # Calcul des coordonnées brutes du trapèze
    emplanture_haut_y = a_sup * col_min + b_sup
    emplanture_bas_y = a_inf * col_min + b_inf
    saumon_haut_y = a_sup * col_max + b_sup
    saumon_bas_y = a_inf * col_max + b_inf
    
    # --- LOGIQUE DU SEUIL TRIANGLE ---
    hauteur_emplanture = abs(emplanture_haut_y - emplanture_bas_y)
    hauteur_saumon = abs(saumon_haut_y - saumon_bas_y)
    
    # Si le saumon est trop petit par rapport à l'emplanture, on force un triangle
    if hauteur_emplanture > 0 and (hauteur_saumon / hauteur_emplanture) < seuil_triangle:
        moyenne_saumon = (saumon_haut_y + saumon_bas_y) / 2
        saumon_haut_y = moyenne_saumon
        saumon_bas_y = moyenne_saumon

    # Construction du polygone final
    coords_x = [col_min, col_max, col_max, col_min]
    coords_y = [emplanture_haut_y, saumon_haut_y, saumon_bas_y, emplanture_bas_y]
    
    return coords_x, coords_y


def selectionner_meilleur_aileron(aileron_gauche, aileron_droit):
    """
    Optimise les deux ailerons rapidement, puis calcule l'erreur complète
    pour sélectionner le meilleur.
    
    Args:
        aileron_gauche, aileron_droit: masques des deux ailerons
    
    Returns:
        tuple: (meilleur_aileron, coords_x, coords_y, nom, erreur)
    """
    coords_x_g, coords_y_g = approximer_aileron_polygone(aileron_gauche)
    coords_x_d, coords_y_d = approximer_aileron_polygone(aileron_droit)
    
    # Maintenant calculer l'erreur complète pour choisir
    erreur_g = calculer_erreur_polygone_complete(aileron_gauche, coords_x_g, coords_y_g)
    erreur_d = calculer_erreur_polygone_complete(aileron_droit, coords_x_d, coords_y_d)
    
    if erreur_g <= erreur_d:
        return aileron_gauche, coords_x_g, coords_y_g, "gauche", erreur_g
    else:
        return aileron_droit, coords_x_d, coords_y_d, "droit", erreur_d

def calculer_dimensions_aileron(coords_x, coords_y):
    """
    Calcule les dimensions d'un aileron à partir des coordonnées du polygone.
    
    - Emplanture (m) : hauteur du bord gauche (toujours vertical)
    - Envergure (E) : distance horizontale entre emplanture et saumon
    - Flèche (f) : décalage horizontal entre le point haut de l'emplanture 
                   et le point haut du saumon
                   NEGATIVE si le saumon est plus haut que l'emplanture
                   POSITIVE si le saumon est plus bas que l'emplanture
    - Saumon (n) : hauteur du bord droit (0 pour un triangle)
    
    Convention:
    - Emplanture = bord gauche (col_min)
    - Saumon = bord droit (col_max)
    - Avec origin='lower', y croissant = vers le haut
    
    Args:
        coords_x, coords_y: listes des coordonnées des sommets du polygone
    
    Returns:
        dict: {'emplanture': float, 'envergure': float, 'fleche': float, 'saumon': float}
    """
    if len(coords_x) == 0:
        return {'emplanture': 0, 'envergure': 0, 'fleche': 0, 'saumon': 0}
    
    # Convertir en arrays numpy pour faciliter les calculs
    coords_x = np.array(coords_x)
    coords_y = np.array(coords_y)
    
    # Identifier la colonne min (emplanture) et max (saumon)
    col_min = coords_x.min()
    col_max = coords_x.max()
    
    # ENVERGURE : distance horizontale entre emplanture et saumon
    envergure = col_max - col_min
    
    # Trouver les points sur l'emplanture (col = col_min)
    mask_emplanture = coords_x == col_min
    points_emplanture_y = coords_y[mask_emplanture]
    
    if len(points_emplanture_y) >= 2:
        # Trapèze : 2 points sur l'emplanture
        emplanture_haut_y = points_emplanture_y.max()  # max car y croissant = vers le haut
        emplanture_bas_y = points_emplanture_y.min()
    else:
        # Triangle : 1 seul point répété sur l'emplanture
        # Les deux premiers points sont sur l'emplanture
        emplanture_haut_y = max(coords_y[0], coords_y[2] if len(coords_y) > 2 else coords_y[0])
        emplanture_bas_y = min(coords_y[0], coords_y[2] if len(coords_y) > 2 else coords_y[0])
    
    # EMPLANTURE : hauteur du bord gauche
    emplanture = abs(emplanture_haut_y - emplanture_bas_y)
    
    # Trouver les points sur le saumon (col = col_max)
    mask_saumon = coords_x == col_max
    points_saumon_y = coords_y[mask_saumon]
    
    if len(points_saumon_y) >= 2:
        # Trapèze : 2 points sur le saumon
        saumon_haut_y = points_saumon_y.max()  # max car y croissant = vers le haut
        saumon_bas_y = points_saumon_y.min()
        saumon = abs(saumon_haut_y - saumon_bas_y)
    else:
        # Triangle : 1 seul point sur le saumon
        saumon = 0
        saumon_haut_y = points_saumon_y[0] if len(points_saumon_y) > 0 else emplanture_haut_y
    
    # FLÈCHE : décalage vertical entre position haute de l'emplanture et position haute du saumon
    # NEGATIVE si saumon_haut_y > emplanture_haut_y (saumon plus haut)
    # POSITIVE si saumon_haut_y < emplanture_haut_y (saumon plus bas)
    fleche = emplanture_haut_y - saumon_haut_y
    
    return emplanture, envergure, fleche, saumon


# Ajouter cette fonction au code principal pour l'affichage
def afficher_dimensions(coords_x, coords_y, affichage=False):
    """
    Affiche les dimensions de l'aileron de manière lisible.
    """
    emplanture, envergure, fleche, saumon = calculer_dimensions_aileron(coords_x, coords_y)
    
    if affichage:
        print("\n" + "="*50)
        print("DIMENSIONS DE L'AILERON (en pixels)")
        print("="*50)
        print(f"Emplanture (m) : {emplanture:.2f} px")
        print(f"Envergure  (E) : {envergure:.2f} px")
        print(f"Flèche     (f) : {fleche:.2f} px")
        print(f"Saumon     (n) : {saumon:.2f} px")
        print("="*50)
    
    return emplanture, envergure, fleche, saumon

def ailerons(data, affichage=False):
    data = np.flip(data, 0)
    
    resultat = filtrer_ailerons(data)
    
    # Séparer les ailerons
    aileron_gauche, aileron_droit = separer_ailerons(resultat)
    
    # Sélectionner le meilleur aileron
    meilleur_aileron, coords_x, coords_y, nom, erreur = selectionner_meilleur_aileron(aileron_gauche, aileron_droit)
    
    if affichage:
        # Afficher le meilleur aileron
        plt.figure(figsize=(10, 8))
        plt.imshow(meilleur_aileron, origin='lower')
        if coords_x:
            coords_x_plot = coords_x + [coords_x[0]]
            coords_y_plot = coords_y + [coords_y[0]]
            plt.plot(coords_x_plot, coords_y_plot, 'r-', linewidth=3, label='Polygone optimisé')
            plt.plot(coords_x, coords_y, 'ro', markersize=10, label='Sommets')
        plt.title(f"Meilleur aileron ({nom}) - Polygone à {len(coords_x)} côtés - Erreur: {erreur} pixels")
        plt.legend()  
        plt.show()

    dimensions = afficher_dimensions(coords_x, coords_y, affichage)
    
    return dimensions
    

# Exemple d'utilisation
if __name__ == "__main__":

    fusee = 9
    data = np.loadtxt(f"AILERONS/fusee_{fusee}.txt")
    emplanture, envergure, fleche, saumon = ailerons(data, True)