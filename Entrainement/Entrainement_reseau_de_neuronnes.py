#%% LIBRAIRIES

import os
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import shutil

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

#%% ETAPE 1 : REDIMENSIONNEMENT DES IMAGES ET MASQUES ORIGINAUX

# DOSSIERS POUR LES FUSEES

# source_img_dir = "./FUSEES/Images_fusees"                      # Dossier source des images de fusées
# source_mask_dir = "./FUSEES/Masques_fusees"                    # Dossier source des masques de fusées
# target_img_dir = "./FUSEES/Banques_images_rn_resized/images"   # Dossier de destination pour les images redimensionnées
# target_mask_dir = "./FUSEES/Banques_images_rn_resized/masks"   # Dossier de destination pour les masques redimensionnés


# DOSSIERS POUR LES MIRES

source_img_dir = "./MIRES/Images_mires"                     # Dossier source des images de mires
source_mask_dir = "./MIRES/Masques_mires"                   # Dossier source des masques de mires
target_img_dir = "./MIRES/Banques_images_rn_resized/images" # Dossier de destination pour les images redimensionnées
target_mask_dir = "./MIRES/Banques_images_rn_resized/masks" # Dossier de destination pour les masques redimensionnés



# Ici on vient vérifier et supprimer le dossier d'images de destination s'il existe déjà 
# pour éviter de rajouter des images dans des bases de données préexistante (ce qui fauserait les résutlats)
if os.path.exists(target_img_dir):
    shutil.rmtree(target_img_dir)
    print(f"Dossier existant supprimé : {target_img_dir}")

# Même raisonnement pour la base de données des masques on vérifie et supprime le dossier s'il existe
if os.path.exists(target_mask_dir):
    shutil.rmtree(target_mask_dir)
    print(f"Dossier existant supprimé : {target_mask_dir}")


os.makedirs(target_img_dir, exist_ok=True) # On crée un dossier pour les images redimensionnées 
os.makedirs(target_mask_dir, exist_ok=True)# On crée un dossier pour les masques redimensionnées 

target_size = (512, 512)   # Taille des images redimensionnees

for fname in os.listdir(source_img_dir): # Ici on parcours toutes les images dans le dossier source des images

    img = Image.open(os.path.join(source_img_dir, fname)).convert("RGB") # On "ouvre" l'image sous format RGB 
    img_resized = img.resize(target_size, Image.BILINEAR)                # On la redimensionne avec une interpolation bilinéaire pour eviter un effet de pixelisation
    img_resized.save(os.path.join(target_img_dir, fname))                # Enfin on la sauvegarde dans le dossier de destination avec le même nom

    base_name = os.path.splitext(fname)[0]      # Ici on vient extraire le nom de base sans l'extension des images (ex: "image_fusee(1).jpg" devient "image_fusee(1)")

    mask_name = base_name.replace("_fusee", "_mask") + ".png"       # Ici on vient construire le squelette des noms des masques correspondant (ex: "image_fusee(1)" devient "image_mask(1)")
    mask_path = os.path.join(source_mask_dir, mask_name)            # On récupère le chemin menant au masque
    mask = Image.open(mask_path)                                    # On "ouvre" le masque
    mask_resized = mask.resize(target_size, Image.NEAREST)          # On la redimensionne avec une interpolation NEAREST pour conserver les labels
    mask_save_name = base_name + "_mask.png"                        # Construction du nom standardisé pour le masque de sortie
    mask_resized.save(os.path.join(target_mask_dir, mask_save_name))# Enfin on sauvegarde le masque redimensionné avec le nouveau nom standardisé

print("Toutes les images et masques ont été redimensionnés")

# Remarque :
# attention ici les images peuvent en effet être bien convertie et redimensionne, 
# mais il est important de les verifier une a une la correspondance masque / image car en effet certaines imagees
# prises depuis certains telephones on des orientations particuliere stockées dans les métadonnées EXIF il faut donc les prendre en compte et les corriger



#%% ETAPE 2 : AUGMENTATION DES DONNEES



# On récupère dans un premier temps le chemin vers le dossier contenant les images redimensionnées

# img_dir = "./FUSEES/Banques_images_rn_resized/images" # Chemin pour les fusées 
img_dir = "./MIRES/Banques_images_rn_resized/images"    # Chemin pour les mires


all_images = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg"))]# Ici on vient récupérer toutes les images dans le dossier redimensionnées

# Division des données en ensembles d'entraînement et de validation
train_imgs, val_imgs = train_test_split(
    all_images,
    test_size=0.2,      # 20% pour la validation
    random_state=20,    # Graine aléatoire pour reproductibilité
    shuffle=True        # On mélange des données avant le split
)


# On définit ici le pipeline de transformations pour l'augmentation des données

transform = A.Compose([

    A.Affine(
        translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},  # Translation verticale et horizontale de plus ou moins 15%
        scale=(0.7, 1.3),         # Zoom entre 70% et 130% pour simuler une fusée plus proche/lointaine
        rotate=(-15, 15),         # Rotation de plus ou moins 15° pour simuler une légère inclinaison du aux fusées mal équilibrées ou encore une mauvaise inclinaison de l'appareil photo
        p=0.8,                    # probabilité d'appliquer cette transformation aux images est de 80%
    ),
    
    A.RandomBrightnessContrast(
        brightness_limit=0.3,     # Variation de la luminosité de plus ou moins 30%
        contrast_limit=0.3,       # Variation des contrastes de l'images de plus ou moins 30%
        p=0.6                     # probabilité d'appliquer cette transformation aux images est de 60%
    ),
    
    
], additional_targets={'mask': 'mask'})# On applique les mêmes transformations aux masques pour garder la bonne correspondance


# DOSSIERS POUR LES FUSEES

# source_img_dir = "./FUSEES/Banques_images_rn_resized/images"    # Dossier source des images des fusees redimensionnes
# source_mask_dir = "./FUSEES/Banques_images_rn_resized/masks"    # Dossier source des masques des fusees redimensionnes
# aug_img_dir = "./FUSEES/Banques_images_aug/images"              # Dossier de destination des images de fusees augmentées
# aug_mask_dir = "./FUSEES/Banques_images_aug/masks"              # Dossier de destination des masques de fusees augmentés


# DOSSIERS POUR LES MIRES

source_img_dir = "./MIRES/Banques_images_rn_resized/images"      # Dossier source des images des mires redimensionnes
source_mask_dir = "./MIRES/Banques_images_rn_resized/masks"      # Dossier source des masques des mires redimensionnes
aug_img_dir = "./MIRES/Banques_images_aug/images"                # Dossier de destination des images de mires augmentées
aug_mask_dir = "./MIRES/Banques_images_aug/masks"                # Dossier de destination des masques de mirees augmentés


# Même raisonnement que précédemmment vérifie et supprime les dossier s'ils existent déjà 
# pour éviter des problèmes de rajout de données a des bases données existantes
if os.path.exists(aug_img_dir):
    shutil.rmtree(aug_img_dir)
    print(f" Dossier existant supprimé : {aug_img_dir}")
if os.path.exists(aug_mask_dir):
    shutil.rmtree(aug_mask_dir)
    print(f"  Dossier existant supprimé : {aug_mask_dir}")


os.makedirs(aug_img_dir, exist_ok=True) # On crée un dossier pour les images augmentées
os.makedirs(aug_mask_dir, exist_ok=True)# On crée un dossier pour les masques augmentés


num_aug = 20 # Nombre d'augmentation par images

# Ici on vient parcourir UNIQUEMENT les images du train set
for fname in train_imgs:
    
    img = np.array(Image.open(os.path.join(source_img_dir, fname)).convert("RGB")) # On "ouvre" l'image sous format RGB et numpy
    mask_name = os.path.splitext(fname)[0] + "_mask.png"                           # On construit le nom du masque correspondant (ex:  fusee = "image_fusee(1).png"  et masque = "image_fusee(1)_mask.png")
    mask = np.array(Image.open(os.path.join(source_mask_dir, mask_name)))          # On charge le masque en array numpy
    
    base = os.path.splitext(fname)[0]    # On extrait comme précédemennt le nom de base sans extension

    Image.fromarray(img).save(os.path.join(aug_img_dir, f"{base}_aug0.png"))        # On sauvegarde l'image originale avec le suffixe "_aug0"
    Image.fromarray(mask).save(os.path.join(aug_mask_dir, f"{base}_aug0_mask.png")) # On sauvegarde le masque original avec le suffixe "_aug0_mask"
    
    for i in range(1, num_aug + 1):    # Boucle pour générer num_aug versions augmentées
    
        augmented = transform(image=img, mask=mask)        # Application des transformations définies précédemment sur l'image ET le masque
        img_aug = augmented['image']        # On récupére l'image augmentée
        mask_aug = augmented['mask']        # On récupére le masques augmenté avec les mêmes transformation géométrique que l'image
        Image.fromarray(img_aug).save(os.path.join(aug_img_dir, f"{base}_aug{i}.png"))        # On sauvegarde l'image augmentée 
        Image.fromarray(mask_aug).save(os.path.join(aug_mask_dir, f"{base}_aug{i}_mask.png")) # On sauvegarde le masque augmentée 


print(" Augmentation terminée images générées")



#%% ETAPE 3 : CREATION DES DATASETS

class FuseeDataset(Dataset):
    
    """
    Cette classe permet de charger automatiquement les paires image/masque 
    depuis les dossiers, de les prétraiter et les fournir au modèle PyTorch 
    pour l'entraînement.
    """
    
    
    def __init__(self, img_dir, mask_dir, image_list):
        
        """
        Cette fonction permet l'initialisation d'un objet de la classe FuseeDataset
        
        Arguments :
            img_dir : chemin vers le dossier contenant les images
            mask_dir : chemin vers le dossier contenant les masques
            image_list : liste des noms de fichiers d'images à charger
        """
        
        self.img_dir = img_dir          # Stockage du chemin des images
        self.mask_dir = mask_dir        # Stockage du chemin des masques
        self.images = image_list        # Stockage de la liste des fichiers
    
    
    def __len__(self):
        
        """
        Cette fonction retourne le nombre total d'images dans le dataset
        """
        
        return len(self.images)
    
    
    def __getitem__(self, idx):
        
        """
        Cette fonction permet de prétraiter une paire image/masque à l'index donné.
        
        Arguments :
            idx : index de l'élément à récupérer
            
        Retourne :
            image : tensor normalisé de forme (3, H, W)
            mask : tensor d'entiers de forme (H, W) 
        """
    
        img_name = self.images[idx]  # Récupération du nom de l'image à l'index idx
        
        mask_name = os.path.splitext(img_name)[0] + "_mask.png" # On construit le nom du masque

        img_path = os.path.join(self.img_dir, img_name)   # chemin vers l'image idx
        mask_path = os.path.join(self.mask_dir, mask_name)# chemin vers le masque idx
        
        image = Image.open(img_path).convert("RGB")       # On récupère l'image en format RBG        
        image = np.array(image, dtype=np.float32) / 255.0 # on normalise les valeurs entre 0 et 1
        image = torch.tensor(image).permute(2, 0, 1)# Enfin on réoragnise et converti l'image pour qu'elle soit compatible avec PyTorch
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) # Moyenne RGB d'ImageNet
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)  # Écart-type RGB d'ImageNet
        image = (image - mean) / std # On normalise l'image : on la centre réduit
        
        # REMARQUE : Ici on utilise donc les moyenne et ecartype d'image NEt car on utilise ici au depart 
        # un réseau deja entraine, il est donc necessaire de lui donner des images en entree coherentes 
        # avec son entrainement initial
        
        mask = Image.open(mask_path)          # On récupère le masque
        mask = np.array(mask, dtype=np.int64) # On le conerti en numpy
        mask = torch.tensor(mask, dtype=torch.long)# On le converti finalement en torch (pour l'entrainment sou PyTorch), toujours en entier
        
        return image, mask  



train_imgs_augmented = [f for f in os.listdir(aug_img_dir) if f.lower().endswith((".png", ".jpg"))]# On récupère la liste de tous les images du dossier d'images augmentées



# CREATION DU DATALOADER D'ENTRAINEMENT

train_dataset = FuseeDataset(# Création du dataset d'entrainement
    aug_img_dir,             # Chemin contenant le dossier des images augmentées
    aug_mask_dir,            # Chemin contenant le dossier des masques augmentées
    train_imgs_augmented     # Liste de toutes les images augmentées
)

train_loader = DataLoader(
    train_dataset,     # Dataset d'entraînement qui correspond ici aux images augmentées
    batch_size=8,      # Nombre d'images par batch pour l'entrainement
    shuffle=True       # Ici on mélange les données à chaque epoch pour éviter l'overfitting
)




# CREATION DU DATALOADER DE VALIDATION

val_dataset = FuseeDataset(# Création du dataset de validation
    source_img_dir,   # Dossier contenant les images redimensionnées 
    source_mask_dir,  # Dossier contenant les masques redimensionnés
    val_imgs          # Liste des images de validation défini précédement
)


val_loader = DataLoader(
    val_dataset,       # Dataset de validation qui correspond aux images originales redimensionnes dediees a la validation, sans augmentations
    batch_size=8,      # Nombre d'images par batch pour le test
    shuffle=False      # Ici on ne fait pas de mélange car on veut suivre l'évolution des prédictions et donc on applique le réseau sur les mêmes images
)


print("\nDatasets créés")


#%% ETAPE 4 : CREATION DU MODELE U-NET

# Ici on crée notre modèle en définissant son architecture et ses paramètres
model = smp.Unet(
    encoder_name="resnet34",         # Encodeur : ResNet34
    encoder_weights="imagenet",      # Poids pré-entraînés sur ImageNet, pour avoir un entrainement plus rapide et efficace
    in_channels=3,                   # Entrée : image RGB (3 canaux)
    
    #classes=4                       # Sortie : 4 classes (fond, fuselage, coiffe, ailerons),
    classes=2                        # Sortie : 2 classes (fonds, mires), 
)

# On utilise GPU (CUDA) si disponible pour de meilleure performances d'entrainement (plus rapide), sinon CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)# On déplace tous les paramètres et buffers du modèle vers le GPU ou CPU
print(f"Device: {device}")

#%% ETAPE 5 : ENTRAÎNEMENT DU MODELE



def compute_class_weights(train_loader, num_classes=4, device='cuda'):
    
    """
    Cette fonction permet de calculer les poids des classes inversement proportionnels à leur fréquence.
    Ainsi Les classes rares (comme les ailerons ou la coiffe coiffe) auront un poids plus élevé et donc une 
    plus grande importance lors de l'entrainement du réseau de neurones
    
    Arguments :
        
       train_loader : torch.utils.data.DataLoader
                      DataLoader contenant les données d'entraînement
       
       num_classes : int, optionnel (défaut=4)
                     Nombre total de classes dans le problème de segmentation
       
       device : str, optionnel (défaut='cuda')
                Device sur lequel transférer les poids calculés
                
    """

    class_counts = torch.zeros(num_classes) # On initialise ici un tensor pour compter les pixels de chaque classe
    
    # On va parcourir tous les batches du train_loader pour analyser la distribution des classes
    for _, masks in tqdm(train_loader, desc="Analyse distribution"):
        for cls in range(num_classes):                # Pour chaque classe, on compte combien de pixels lui appartiennent
            class_counts[cls] += (masks == cls).sum() # Somme des pixels de la classe cls
    
    total_pixels = class_counts.sum() # Nombre total de pixels dans tous les masques
    class_weights = total_pixels / (class_counts * num_classes + 1e-6) # Détermination des poids de chaque classe

    return class_weights.to(device)







class EarlyStopping:
    
    """
    Cette classe permet l'arrêt de l'entrainement du réseau de neurones automatiquement.
    Si la validation loss ne s'améliore plus pendant plusieurs epochs consécutifs l'entrainement s'arrête.
    """
    
    def __init__(self, patience=7, min_delta=0.001):
        
        """
        Cette fonction permet l'initialisation d'un objet de la classe EarlyStopping
        
        Args:
            patience: Nombre d'epochs à attendre sans amélioration avant d'arrêter l'entrainement
            min_delta: Amélioration minimale pour considérer un progrès entre l'epoch actuelle
                       et celle qui a donné la meilleure validation loss        
        """
        
        self.patience = patience      # Nombre d'epochs de tolérance sans améliorations
        self.min_delta = min_delta    # Seuil d'amélioration minimal
        self.counter = 0              # Compteur d'epochs sans amélioration
        self.best_loss = None         # Meilleure validation loss observée
        self.early_stop = False       # Flag d'arrêt, si il vaut True, alors l'entrainement s'arrete
        
    def __call__(self, val_loss):
        
        """
        Cette fonction permet de vérifier si l'entraînement doit s'arrêter.
        
        Arguments :
            val_loss : Loss de validation de l'epoch actuelle
        """
        
        if self.best_loss is None:  # On initialise lors du premier Appel
            self.best_loss = val_loss # On stocke la vlaeur de validation loss
            
        elif val_loss > self.best_loss - self.min_delta: # Si la validation loss ne s'améliore pas
            self.counter += 1 # On incremente 1 au compteur
            print(f" EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:  # Si on atteint la patience, on déclenche l'arrêt
                self.early_stop = True
                
        else: # Sinon on stocke la valeur de la loss de validation et on remet a zero le compteur.
            self.best_loss = val_loss
            self.counter = 0






def visualize_predictions(model, val_loader, device, epoch, num_samples=4):
    
    """
    Cette fonction permet l'affichage et la sauvegarde de quelques prédictions pour vérifier visuellement
    la progression de l'entraînement au fil des epoch
    
    Arguments :
        
       model : torch.nn.Module
               Modèle U-Net entraîné à évaluer
       
       val_loader : torch.utils.data.DataLoader
                    DataLoader contenant les images de validation
       
       device : torch.device
                Device utilisé pour le calcul (CPU ou CUDA/GPU)
       
       epoch : int
               Numéro de l'epoch actuelle 
               
       num_samples : int, optionnel (défaut=4)
                      Nombre total de classes dans le problème de segmentation
    """
    
    model.eval() # On récupere le model en mode evalutaion ( donc pour effectuer seulement la prediction)
    
    imgs, masks = next(iter(val_loader))           # On prend le premier batch de validation
    imgs, masks = imgs.to(device), masks.to(device)# puis on le transfert vers GPU/CPU
    

    with torch.no_grad():# On désactive le calcul des gradients pour économiser la mémoire
        outputs = model(imgs) # On passe le batch dans le forward pass (prédictions brutes, en logits)
        preds = torch.argmax(outputs, dim=1)# Conversion en labels de classe avec argmax
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples*3))# On crée une figure
    
    for i in range(min(num_samples, len(imgs))): # On va venir chacun des résultats
                
        img_np = imgs[i].cpu().permute(1,2,0).numpy()  # On converti l'image en numpy
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]) # On dénormailise l'image
        img_np = np.clip(img_np, 0, 1)  # Finalement on "clip" l'image pour garantir des valeurs entre 0 et 1
                
        axes[i,0].imshow(img_np)# Affichage de l'image originale
        axes[i,0].set_title("Image")
        axes[i,0].axis('off')
        
        axes[i,1].imshow(masks[i].cpu(), cmap='gray', vmin=0, vmax=1)# Affichage du masque défini manuellement
        axes[i,1].set_title("Ground Truth")
        axes[i,1].axis('off')
        
        axes[i,2].imshow(preds[i].cpu(), cmap='gray', vmin=0, vmax=1)# Affichage du masque prédit par le réseau de neurones
        axes[i,2].set_title("Prédiction")
        axes[i,2].axis('off')
    
    plt.suptitle(f"Epoch {epoch}", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(f"predictions_epoch{epoch:03d}.png", dpi=100, bbox_inches='tight') # Sauvegarde des résultats en indiquant le nombre d'epoch
    plt.close()





def compute_metrics(outputs, masks, num_classes=4):
    
    """
    Cette fonction permet de calculer le score Dice et IoU pour un batch multiclasses.
    Ces scores permettent au fil des itérations d'observer la performance du réseau de neurones
    
    Arguments :
        
        outputs : Tensor 
                 Prédictions brutes du modèle (logits) pour chaque classe et chaque pixel
        
        masks : Tensor 
                Masque d'entrainement (réalisé manuellement) contenant les labels de classe pour chaque pixel
        
        num_classes : int, optionnel (défaut=4)
                      Nombre total de classes dans le problème de segmentation
    
    """
    
    preds = torch.argmax(outputs, dim=1)# On converti les logits en labels de classe avec la fonction argmax

    dice_total = 0.0
    iou_total  = 0.0

    for cls in range(1, num_classes): # On effectue tout les calculs suivant sauf pour la classe du fonds.
        
        pred_cls = (preds == cls).float() # Prédictions de la classe cls
        mask_cls = (masks == cls).float() # Vérité terrain de la classe cls

        intersection = (pred_cls * mask_cls).sum() # On calcul ici l'intersection des deux masques : soit les pixels prédits ET réels pour cette classe
        union = pred_cls.sum() + mask_cls.sum()    # On calcul ici l'union des deux masques : soit la somme totale des pixels prédits + réels pour cette classe

        dice = (2.0 * intersection) / (union + 1e-7) # Formule du Dice
        dice_total += dice

        iou = intersection / (pred_cls.sum() + mask_cls.sum() - intersection + 1e-7) # Formule de l'IoU
        iou_total += iou

    dice_mean = dice_total / (num_classes - 1) # On effectue la moyenne Dice pour avoir un score moyen entre 0 et 1
    iou_mean  = iou_total / (num_classes - 1)  # On effectue la moyenne IoU pour avoir un score moyen entre 0 et 1

    return dice_mean.item(), iou_mean.item()



# APPEL DE LA FONCTION COMPUTE_CLASS_WEIGHTS pour compenser le déséquilibre des classes

#class_weights = compute_class_weights(train_loader, num_classes=4, device=device)  # Pour fusées
class_weights = compute_class_weights(train_loader, num_classes=2, device=device)   # Pour mires 

criterion = nn.CrossEntropyLoss(weight=class_weights) # On définit ici la fonction de perte pour l'entrainement du réseau, ici la fonction crossentropy avec les poids des classses


optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau( # Ce "Scheduler" permet de réduire le learning rate si la validation loss ne s'améliore pas
    optimizer,
    mode='min',        # On minimise ici la loss
    factor=0.5,        # Réduction du LR de 50% à chaque plateau
    patience=3,        # On attend 3 epochs sans amélioration avant de réduire le learning rate
    min_lr=1e-7,       # learning rate minimum 
)

early_stopping = EarlyStopping(patience=7, min_delta=0.001)# Création d'un objet EarlyStopping pour arreter l'entrainement au bout de 7 epoch sans amelioration
epochs = 50   # Nombre maximal d'epoch pour l'entrainement

# NOMBRE DE CLASSES
#num_classes = 4 # Pour les fusées
num_classes = 2  # Pour les mires

train_losses = []  # Historique des losses d'entraînement
val_losses   = []  # Historique des losses de validation
val_dices    = []  # Historique des scores Dice
val_ious     = []  # Historique des scores IoU

best_val_loss = float('inf')# On initialise ici la val loss a l'infini
best_epoch = 0

for epoch in range(epochs):

    model.train() # Passage du model en mode netrainement
    train_loss = 0.0

    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]"): # On parcours ici tous les batches du train_loader

        imgs, masks = imgs.to(device), masks.to(device) # On transfert les calculs vers le GPU/CPU
        optimizer.zero_grad()                           # Premierement on remet à zéro des gradient
        outputs = model(imgs)                           # Puis on fait passer l'image dans le Forward pass : prédictions du modèle
        loss = criterion(outputs, masks)                # On calcul de la loss
        loss.backward()                                 # Grace a la loss, on peut effectuer la Backpropagation ce qui nous permet de calculer les gradients
        optimizer.step()                                # Avec ces gradients on met à jour des poids du modèle
        train_loss += loss.item()                       # Enfin on accumule la loss

    train_loss /= len(train_loader)    # Calcul de la loss moyenne sur tous les batches
    train_losses.append(train_loss)    # Stockage de la valeur dans l'historique


    model.eval()      # Passage en mode évaluation
    val_loss = 0.0    # On initialise la loss de validation a 0
    dice_scores = []  # Liste des scores Dice de chaque batch
    iou_scores  = []  # Liste des scores IoU de chaque batch


    with torch.no_grad(): # On désactivation du calcul des gradients (pour économiser de la mémoire)        
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VAL]"): # On parcour ici tous les batches du val_loader
            
            imgs, masks = imgs.to(device), masks.to(device)  # Comme précédemment on transfert les calculs vers le GPU/CPU
            outputs = model(imgs)                            # Puis on effectue les prédictions
            loss = criterion(outputs, masks)                 # Ce qui nous permet d'obtenir la loss 
            val_loss += loss.item()                          # et de l'accumuler
            dice, iou = compute_metrics(outputs, masks)      # On calcul le score Dive et IoU et on les stocke
            dice_scores.append(dice)
            iou_scores.append(iou)

    
    val_loss /= len(val_loader)              # Calcul de la loss de validation moyenne sur tous les batches
    mean_dice = float(np.mean(dice_scores))  # Calcul du score Dice moyen sur tous les batches
    mean_iou  = float(np.mean(iou_scores))   # Calcul du score IoU moyen sur tous les batches

    val_losses.append(val_loss) # On sauvegarde ces données dans les listes
    val_dices.append(mean_dice)
    val_ious.append(mean_iou)

    scheduler.step(val_loss)  # On ajuste le LR en fonction de la val_loss
    current_lr = optimizer.param_groups[0]['lr']  # Et on récupération du LR actuel

    if val_loss < best_val_loss:    # Si la loss de validation s'est améliorée
        best_val_loss = val_loss # On stocke la meilleure loss de validation
        best_epoch = epoch + 1   # On stocke l'epoch
        
        torch.save({ # On sauvegarde ce nouveau réseau correspond théoriquement au meilleur (car minimise la loss de validation)
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_dice': mean_dice,
            'val_iou': mean_iou,
            'class_weights': class_weights,
        }, "modele_fusee_BEST.pth")
        


    print( # Affichage des métriques de l'epoch
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Dice: {mean_dice:.4f} | "
        f"IoU: {mean_iou:.4f} | "
        f"LR: {current_lr:.2e}"
    )


    if (epoch + 1) % 5 == 0:  # Tous les 5 epochs
        visualize_predictions(model, val_loader, device, epoch+1) # On visualise les prédictions faite par le réseau a l'aide de la fonction visualize_predictions


    early_stopping(val_loss) # Vérification si on doit arrêter l'entrainement du réseau
    
    if early_stopping.early_stop: # Si early_stop = True
        print(f"\nEarly stopping déclenché à l'epoch {epoch+1}")
        break # On sort de la boucle d'entrainement

torch.save(model.state_dict(), "modele_mires_FINALlll.pth") # On enregistre aussi le dernier réseau de neurones (qui lui minimise la loss d'entrainement)
print("\nModèle final sauvegardé : modele_fusee_FINAL.pth")
print("Meilleur modèle : modele_fusee_BEST.pth (epoch {best_epoch}, val_loss={best_val_loss:.4f})")
