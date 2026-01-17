import numpy as np


def detecter_contour(matrice):
    """Extrait les coordonnées X et Y du contour supérieur d'une forme binaire."""
    # Récupération des dimensions de la matrice (image binaire)
    m, n = matrice.shape
    # Initialisation des listes pour stocker les points du contour
    contour_y = []
    contour_x = []

    # Parcours de chaque colonne (x) pour trouver le premier pixel blanc (y)
    for x in range(n):
        for y in range(m):
            # Si le pixel est considéré comme "blanc" (au-dessus du seuil)
            if matrice[y, x] > 128 / 255:
                # Ajout des coordonnées (m-y pour inverser l'axe Kivy vers mathématique)
                contour_y.append(m - y)
                contour_x.append(n - x)
                break
    # Retourne les tableaux inversés pour correspondre à l'ordre naturel
    return np.flip(contour_x), np.flip(contour_y)


def resize_to_N(arr, N):
    """Redimensionne un tableau de données à une taille N via interpolation linéaire."""
    arr = np.asarray(arr)  # Assure que l'entrée est un tableau numpy
    n = arr.size  # Taille actuelle
    x_old = np.linspace(0, 1, n)  # Vecteur de position original
    x_new = np.linspace(0, 1, N)  # Nouveau vecteur de position
    # Calcule les nouvelles valeurs interpolées
    return np.interp(x_new, x_old, arr)


def calculer_regression_R2(contour_x, contour_y):
    """Calcule le coefficient de détermination R² pour évaluer la linéarité du contour."""
    contour_x = np.array(contour_x)
    contour_y = np.array(contour_y)

    # Fit linéaire (y = ax + b)
    a, b = np.polyfit(contour_x, contour_y, 1)
    y_pred = a * contour_x + b

    # Somme des carrés des résidus
    SS_res = np.sum((contour_y - y_pred) ** 2)
    # Somme totale des carrés
    SS_tot = np.sum((contour_y - np.mean(contour_y)) ** 2)

    # Formule du R² (plus proche de 1 = forme rectiligne/conique)
    R2 = 1 - (SS_res / SS_tot)

    return R2, a


def classifier_arc(matrice, seuil_R2_conique=0.9555, seuil_pointe=0.7256):
    """Classifie la pointe de la fusée en 0 (Parabolique), 1 (Ogivale) ou 2 (Conique)."""
    # Extraction du contour depuis la matrice binaire
    contour_x, contour_y = detecter_contour(matrice)

    # Normalisation à 100 points pour uniformiser le calcul
    contour_x = resize_to_N(contour_x, 100)
    contour_y = resize_to_N(contour_y, 100)

    # Mise à l'échelle des coordonnées entre 0 et 1 (Normalisation spatiale)
    contour_xx = contour_x - np.min(contour_x)
    contour_xx = contour_xx / np.max(contour_xx)

    contour_yy = contour_y - np.min(contour_y)
    contour_yy = contour_yy / np.max(contour_yy)

    # Recherche du sommet (pointe) de l'ogive
    max_y = np.max(contour_yy)
    indices_max = np.where(contour_yy == max_y)[0]
    # Calcul du centre de la pointe si plat
    x_centre = (indices_max[0] + indices_max[-1]) / 2
    idx_pointe = int(x_centre)

    # Séparation du contour en deux flancs : gauche et droit
    x_gauche = contour_xx[: idx_pointe + 1]
    y_gauche = contour_yy[: idx_pointe + 1]

    x_droite = contour_xx[idx_pointe + 1 :]
    y_droite = contour_yy[idx_pointe + 1 :]

    # Calcul de la linéarité sur chaque flanc
    R2_gauche, _ = calculer_regression_R2(x_gauche, y_gauche)
    R2_droite, _ = calculer_regression_R2(x_droite, y_droite)

    # Moyenne de linéarité (si proche de 1, les flancs sont des segments de droite)
    R2_moyenne = (R2_gauche + R2_droite) / 2

    # Si très rectiligne, c'est un cône parfait (Type 2)
    if R2_moyenne > seuil_R2_conique:
        return 2
    else:
        nbp = 20
        d = 12

        # Analyse de la courbure à la pointe pour distinguer Parabolique d'Ogivale
        num_points_gauche_pointe = nbp
        num_points_droite_pointe = nbp

        # Extraction des points situés juste autour du sommet
        x_gauche_pointe = x_gauche[
            idx_pointe - num_points_gauche_pointe + 1 : idx_pointe - (d - 1)
        ]
        y_gauche_pointe = y_gauche[
            idx_pointe - num_points_gauche_pointe + 1 : idx_pointe - (d - 1)
        ]

        x_droite_pointe = x_droite[d:num_points_droite_pointe]
        y_droite_pointe = y_droite[d:num_points_droite_pointe]

        # Calcul de la pente locale des deux côtés de la pointe
        _, a = calculer_regression_R2(x_gauche_pointe, y_gauche_pointe)
        _, b = calculer_regression_R2(x_droite_pointe, y_droite_pointe)

        # Calcul du coefficient angulaire moyen
        coeff_moy = min(abs(a), abs(b))

        # Si la pointe est "aiguë" (pente forte), c'est une ogive (Type 1)
        if coeff_moy > seuil_pointe:
            return 1
    # Sinon, la pointe est arrondie (Type 0)
    return 0


def noseConeType(cone):
    """Point d'entrée principal : convertit l'image en matrice et renvoie le type."""
    # Binarisation de l'image (seuil strict à 240/255)
    matrice = np.where(cone > 240 / 255, 255, 0).astype(np.uint8)
    # Appel de l'algorithme de classification
    type_arc = classifier_arc(matrice)
    # Log de débogage dans la console
    print(f"[NOSE CONE] Type détecté : {type_arc}")
    return type_arc
