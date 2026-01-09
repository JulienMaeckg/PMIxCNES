import os
import cv2
import numpy as np

#%% POUR CREER LE MASQUE

# CETTE FONCTION PERMET TRES SIMPLEMENT DE CREER LES MASQUES DES FUSEES / MIRES
# NORMALEMENT EN SUIVANT LES ETAPES CA SE FAIT TOUT SEUL

def create_multiclass_mask(image_path, mask_output_path, preview_alpha=0.5):
    """
    FONCTION GENEREE PAR IA
    Annotateur interactif multi-classes (1=fuselage, 2=cône, 3=ailerons, 0=fond)
    pour une seule image.

    Commandes :
      - 1,2,3 : changer la classe de dessin
      - clic gauche : dessiner
      - clic droit : effacer
      - + / - : ajuster la taille du pinceau
      - p : activer/désactiver le mode polygone (Enter pour remplir)
      - c : effacer le masque
      - u : annuler la dernière action
      - s : sauvegarder le masque (PNG avec valeurs 0–3)
      - q : quitter sans sauvegarder
    """

    if not os.path.exists(image_path):
        print("Image introuvable :", image_path)
        return

    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("Impossible d'ouvrir l'image :", image_path)
        return

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    colors = {
        1: (0, 0, 255),     # rouge : fuselage
        2: (0, 255, 255),   # jaune : cône
        3: (0, 255, 0)      # vert : ailerons
    }

    brush_size = 10
    current_class = 1
    drawing = False
    erasing = False
    last_point = None
    mask_stack = []
    polygon_mode = False
    polygon_points = []

    window_name = f"Annotation : {os.path.basename(image_path)}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, erasing, last_point, mask_stack, polygon_points
        if polygon_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                polygon_points.append((x, y))
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)
            mask_stack.append(mask.copy())
            cv2.circle(mask, (x, y), brush_size, current_class, -1)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.line(mask, last_point, (x, y), current_class, thickness=brush_size*2)
            last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

        if event == cv2.EVENT_RBUTTONDOWN:
            erasing = True
            last_point = (x, y)
            mask_stack.append(mask.copy())
            cv2.circle(mask, (x, y), brush_size, 0, -1)
        elif event == cv2.EVENT_MOUSEMOVE and erasing:
            cv2.line(mask, last_point, (x, y), 0, thickness=brush_size*2)
            last_point = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            erasing = False

    cv2.setMouseCallback(window_name, on_mouse)

    print("  Instructions :")
    print("  1=fuselage  2=cône  3=ailerons")
    print("  clic gauche = dessiner, clic droit = effacer")
    print("  + / - pour changer la taille du pinceau")
    print("  p pour activer le mode polygone (Enter pour remplir)")
    print("  s pour sauvegarder, q pour quitter")

    while True:
        # colorisation du masque pour affichage
        color_mask = np.zeros_like(img)
        for cls, color in colors.items():
            color_mask[mask == cls] = color

        overlay = cv2.addWeighted(img, 1.0, color_mask, preview_alpha, 0)
        display = overlay.copy()

        cv2.putText(display, f"Classe: {current_class}  Brush:{brush_size}px", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        mode_str = "POLYGON" if polygon_mode else "PAINT"
        cv2.putText(display, f"Mode: {mode_str}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            print("Fermeture sans sauvegarde.")
            break
        elif key == ord('c'):
            mask = np.zeros((h, w), dtype=np.uint8)
            mask_stack = []
            print("Masque effacé.")
        elif key == ord('u'):
            if mask_stack:
                mask = mask_stack.pop()
                print("↩️ Annulé.")
        elif key == ord('+') or key == ord('='):
            brush_size = min(200, brush_size + 1)
        elif key == ord('-') or key == ord('_'):
            brush_size = max(1, brush_size - 1)
        elif key == ord('p'):
            polygon_mode = not polygon_mode
            polygon_points = []
            print("Mode polygone :", polygon_mode)
        elif key == 13 or key == 10:  # Enter pour remplir
            if polygon_mode and len(polygon_points) >= 3:
                mask_stack.append(mask.copy())
                pts = np.array(polygon_points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], current_class)
                polygon_points = []
                print("Polygone rempli.")
        elif key in [ord('1'), ord('2'), ord('3')]:
            current_class = int(chr(key))
            print("Classe courante :", current_class)
        elif key == ord('s'):
            _, buf = cv2.imencode(".png", mask)
            buf.tofile(mask_output_path)
            print("Masque sauvegardé :", mask_output_path)
            break

    cv2.destroyAllWindows()


create_multiclass_mask("./Banque_images/TEST3.jpg", "./Banque_images/TEST3_mask.jpg")
# LE PREMIER CHEMIN MENE VERS LIMAGE DONT ON VEUT REALISER LE MASQUE
# LE DEUXIEME CHEMIN MENE VERS L'ENDROIT ET LE NOM DE L'IMAGE DU MASQUE QUE L'ON SOUHAITE CREER
#%% POUR AFFICHER LE MASQUE

# CETTE FONCTION SERT PRINCIPALEMENT A VERIFIER QUE LE MASQUE A BIEN ETE CREE

def afficher_mask_multiclass(image_path, mask_path, facteur=0.5):
    """
    FONCTION GENEREE PAR IA
    Affiche le masque multi-classes en taille réduite avec couleurs distinctes :
      0 = fond (noir)
      1 = fuselage (rouge)
      2 = cône (jaune)
      3 = ailerons (vert)

    Paramètres :
      image_path : chemin de l'image d'origine
      mask_path : chemin du masque (PNG multi-classes)
      facteur : facteur d'échelle pour réduire l'affichage (ex : 0.5 = moitié)
    """

    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    if img is None or mask is None:
        print("Erreur de chargement d'image ou de masque")
        return

    if img.shape[:2] != mask.shape[:2]:
        print("Dimensions différentes entre image et masque. Redimensionnement du masque.")
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    color_mask = np.zeros_like(img)
    color_mask[mask == 1] = (0, 0, 255)     # rouge : fuselage
    color_mask[mask == 2] = (0, 255, 255)   # jaune : cône
    color_mask[mask == 3] = (0, 255, 0)     # vert : ailerons

    overlay = cv2.addWeighted(img, 0.6, color_mask, 0.4, 0)

    if facteur != 1.0:
        new_size = (int(img.shape[1] * facteur), int(img.shape[0] * facteur))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        color_mask = cv2.resize(color_mask, new_size, interpolation=cv2.INTER_NEAREST)
        overlay = cv2.resize(overlay, new_size, interpolation=cv2.INTER_AREA)

    legende = overlay.copy()
    y0 = 25
    cv2.putText(legende, "Legende :", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.rectangle(legende, (100, y0-15), (120, y0+5), (0,0,255), -1)
    cv2.putText(legende, "Fuselage", (125, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.rectangle(legende, (250, y0-15), (270, y0+5), (0,255,255), -1)
    cv2.putText(legende, "Cone", (275, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.rectangle(legende, (370, y0-15), (390, y0+5), (0,255,0), -1)
    cv2.putText(legende, "Ailerons", (395, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # Affichage
    cv2.imshow("Masque coloré", color_mask)
    cv2.imshow("Image + Masque (overlay + légende)", legende)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#%% TEST
afficher_mask_multiclass(
    "./Banque_images/img8.jpg",   # IMAGE
    "./Banque_images/img8_mask.png",# SON MASQUE
    facteur=0.4
)