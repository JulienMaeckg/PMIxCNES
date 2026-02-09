#%% LIBRAIRIES

import os
import cv2
import numpy as np

#%% FONCTION

def Creation_mask(image_path, mask_path, alpha=0.5):
    """
    
    Cette fonctions permet de créer manuellement les classes à destination de l'entrainement du réseau de neuronnes.
    Le masque générer sera en format .png aux dimensions de l'image orginale avec des valeurs en fonction des classes dessinées.


    Argument :
        
        image_path : chemin de l'image d'origine sur laquelle on veut dessiner un masque
        mask_output_path : chemin d'enregistrement de l'image en format .png
        alpha : taux de transparence entre l'image et le masque dessine

    
    Commandes interne au lancement de la fonction:
        
      - touche 1,2,3   : changer la classe de dessin
      - clic gauche    : dessiner
      - clic droit     : effacer
      - touche + et -  : ajuster la taille du pinceau
      - touche u       : annuler la modification
      - touche c       : pour effacer toutes les classes
      - touhce s       : sauvegarder le masque
      - touche q       : quitter sans sauvegarder
    """

    if not os.path.exists(image_path): # DEBUG pour savoir si l'image est trouvable ou non
        print("Image introuvable :", image_path)
        return

    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    if img is None:         # DEBUG pour savoir si l'image est ouvrable
        print("Impossible d'ouvrir l'image :", image_path)
        return

    h, w = img.shape[:2]     # On récupère ici hauteur et largeur de l'image originale

    mask = np.zeros((h, w), dtype=np.uint8) # On crée un masque vide (que des 0) aux dimensions de l'image originale

    couleurs = {            # Définition des couleurs visibles du masque dessiné sur l'image
        1: (0, 0, 255),     # rouge : fuselage
        2: (0, 255, 255),   # jaune : cône
        3: (0, 255, 0)      # vert : ailerons
    }

    taille_pinceau = 10     # Taille initiale du pinceau en pixels
    current_class = 1       # Classe par défaut, ici le fuselge
    drawing = False         # Bool indiquant si on est en train de dessiner
    erasing = False         # Bool indiquant si on est en train de d'effacer
    last_point = None       # Dernier point dessiné 
    mask_stack = []         # Pile pour stocker l'historique des masques
    
    window_name = f"{os.path.basename(image_path)}" # ici on a le titre de la fenetre
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param): # Cette fonction est appelé lorsque l'on fait un événement avec la souris
        
        nonlocal drawing, erasing, last_point, mask_stack 
       
        
        if event == cv2.EVENT_LBUTTONDOWN:  # Quant on fait un clic gauche
            drawing = True                  # On dessine
            last_point = (x, y)             # On sauvegarde la position 
            mask_stack.append(mask.copy())  # On sauvegarde le masque actuel
            cv2.circle(mask, (x, y), taille_pinceau, current_class, -1)# On dessine des cercle en fonction de la classe sélectionnée
            
        elif event == cv2.EVENT_MOUSEMOVE and drawing: # Quant la souris se déplace avec le clic gauche enfoncé en mode dessin
            cv2.line(mask, last_point, (x, y), current_class, thickness=taille_pinceau*2) # On trace des lignes entre le point initial et actuel
            last_point = (x, y) # On met a jour le point actuel
            
        elif event == cv2.EVENT_LBUTTONUP: # Quant on arrete le clic gauche 
            drawing = False # On arrete le mode dessin



        if event == cv2.EVENT_RBUTTONDOWN: # Quant on fait un clic droit
            erasing = True                 # On efface
            last_point = (x, y)            # On sauvegarde la position 
            mask_stack.append(mask.copy()) # On sauvegarde le masque actuel
            cv2.circle(mask, (x, y), taille_pinceau, 0, -1)# On efface avec des cercle (en mettant la valeur 0)
            
        elif event == cv2.EVENT_MOUSEMOVE and erasing: # Quant la souris se déplace avec le clic gauche enfoncé en mode effacage
            cv2.line(mask, last_point, (x, y), 0, thickness=taille_pinceau*2)# On trace des lignes entre le point initial et actuel
            last_point = (x, y)# On met a jour le point actuel
            
        elif event == cv2.EVENT_RBUTTONUP: # Quant on arrete le clic gauche 
            erasing = False # On arrete le mode effacage

    cv2.setMouseCallback(window_name, on_mouse)    # Ici on enregistre la fonction callback pour les événements souris



    while True: # Ici on fait une boucle infini pour prendre en compte les requêtes utilisateurs
        
        color_mask = np.zeros_like(img)  # Ici on crée un masque coloré de même taille que l'image pour superpose l'image original avec celui la en couleur pour savoir ou est ce que l'on trace

        for cls, color in couleurs.items():
            color_mask[mask == cls] = color

        overlay = cv2.addWeighted(img, 1.0, color_mask, alpha, 0) # On superpose l'image originale et du masque coloré avec transparence
        display = overlay.copy()

        cv2.imshow(window_name, display) # On Affiche ici l'image avec le masque coloré

        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'): # Si on appuie sur q on quitte la fonction sans sauvegarder
            print("Fermeture sans sauvegarde.")
            break
        
        elif key == ord('c'):# Si on appuie sur c on annule toutes les modifications effectues
            mask = np.zeros((h, w), dtype=np.uint8) # on remet a zero le masque
            mask_stack = [] # on reinitialise la pile des masques
            print("Masque effacé.")
        
        elif key == ord('u'):# Si on appui sur u on revient en arrière d'une modification
            if mask_stack: # Si la pile contient au moins une sauvegarde
                mask = mask_stack.pop() # On supprime la dernière
                print("Annulé")
        
        elif key == ord('+') or key == ord('='):# Si on appuie sur + on augmente la taille du pinceau
            taille_pinceau = min(200, taille_pinceau + 1)
        
        elif key == ord('-') or key == ord('_'):# Si on appuie sur - on diminue la taille du pinceau
            taille_pinceau = max(1, taille_pinceau - 1)
        
        
        elif key in [ord('1'), ord('2'), ord('3')]:# Si on appuie sur 1, 2 ou 3 on change de classe
            current_class = int(chr(key))
            print("Classe courante :", current_class)
        
        elif key == ord('s'): # Si on appuie sur s quitte la fonction en sauvegardant sous format png
            _, buf = cv2.imencode(".png", mask)# On sauvegarde ici le masque sous format png
            buf.tofile(mask_path)
            print("Masque sauvegardé :", mask_path)
            break

    cv2.destroyAllWindows() # Si on quitte la boucle True, avec s ou q on détruit toute les fenetres
#%% EXEMPLE D'UTILISATION DE LA FONCTION

Creation_mask("./FUSEES/Images_fusees/image_fusee (31).jpg", "./FUSEES/Images_fusees/image_mask (31).png")
