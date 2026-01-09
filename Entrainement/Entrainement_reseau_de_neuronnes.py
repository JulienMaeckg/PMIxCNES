import os
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#%%
# ════════════════════════════════════════════════════════════════════════════════
# ███████╗████████╗ █████╗ ██████╗ ███████╗     ██╗
# ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝    ███║
# █████╗     ██║   ███████║██████╔╝█████╗      ╚██║
# ██╔══╝     ██║   ██╔══██║██╔═══╝ ██╔══╝       ██║
# ███████╗   ██║   ██║  ██║██║     ███████╗     ██║
# ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚══════╝     ╚═╝
#                                                    
# ════════════════════════════════════════════════════════════════════════════════
#            REDIMENSIONNEMENT DES IMAGES ET MASQUES ORIGINAUX
# ════════════════════════════════════════════════════════════════════════════════
# Les réseaux de neurones nécessitent des images de taille fixe.
# On redimensionne toutes les images et masques à 512x512 pixels.
# ════════════════════════════════════════════════════════════════════════════════

# Dossiers sources des images et masques
source_img_dir = "./fusee/Images"
source_mask_dir = "./fusee/Masques"

# Chemins des dossiers de destination (images et masques redimensionnés)
target_img_dir = "./Banques_images_rn_resized_V2/images"
target_mask_dir = "./Banques_images_rn_resized_V2/masks"

# Création des dossiers si ils n'existent pas
os.makedirs(target_img_dir, exist_ok=True)
os.makedirs(target_mask_dir, exist_ok=True)

# Taille cible pour l'entraînement (largeur x hauteur en pixels)
target_size = (512, 512)  

# On parcours toutes les images du dossier source
for fname in os.listdir(source_img_dir):
    #  On ignore ici les fichiers qui ne sont pas des images
    if not fname.lower().endswith((".jpg", ".png")):
        continue

    # ─────────────────────────────────────────────────────────────────────────
    # Traitement de l'image RGB
    # ─────────────────────────────────────────────────────────────────────────
    img = Image.open(os.path.join(source_img_dir, fname)).convert("RGB")
    img_resized = img.resize(target_size, Image.BILINEAR)
    img_resized.save(os.path.join(target_img_dir, fname))

    # ─────────────────────────────────────────────────────────────────────────
    # Traitement du masque (labels de segmentation)
    # ─────────────────────────────────────────────────────────────────────────
    mask_name = os.path.splitext(fname)[0] + "_mask.png"
    mask = Image.open(os.path.join(source_mask_dir, mask_name))
    mask_resized = mask.resize(target_size, Image.NEAREST)  # NEAREST pour conserver les labels
    mask_resized.save(os.path.join(target_mask_dir, mask_name))

print("Toutes les images et masques ont été redimensionnés et sauvegardés dans :")
print(target_img_dir)
print(target_mask_dir)

#%%
# ════════════════════════════════════════════════════════════════════════════════
# ███████╗████████╗ █████╗ ██████╗ ███████╗    ██████╗ 
# ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝    ╚════██╗
# █████╗     ██║   ███████║██████╔╝█████╗       █████╔╝
# ██╔══╝     ██║   ██╔══██║██╔═══╝ ██╔══╝      ██╔═══╝ 
# ███████╗   ██║   ██║  ██║██║     ███████╗    ███████╗
# ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚══════╝    ╚══════╝
#                                                        
# ════════════════════════════════════════════════════════════════════════════════
#                        AUGMENTATION DE DONNÉES
# ════════════════════════════════════════════════════════════════════════════════
# L'augmentation permet de créer des variations artificielles des images
# pour enrichir le dataset et améliorer la généralisation du modèle.
#
# Transformations appliquées :
#   - Flip horizontal/vertical
#   - Rotation aléatoire (90°, 180°, 270°)
#   - Translation, zoom, rotation libre
#   - Variation de luminosité et contraste
#   - Ajout de bruit gaussien
# ════════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# Configuration du pipeline d'augmentation avec Albumentations
# ─────────────────────────────────────────────────────────────────────────────

transform = A.Compose([
    A.HorizontalFlip(p=0.5),               # Flip horizontal (50% de chance)
    A.VerticalFlip(p=0.3),                 # Flip vertical (30% de chance)
    A.RandomRotate90(p=0.5),               # Rotation 90° (50% de chance)
    A.ShiftScaleRotate(                    # Translation + zoom + rotation libre
        shift_limit=0.1,                   # Translation max 10%
        scale_limit=0.2,                   # Zoom entre 80% et 120%
        rotate_limit=20,                   # Rotation max ±20°
        p=0.7,                             # 70% de chance
        border_mode=cv2.BORDER_CONSTANT    # Remplir les bords en noir
    ),
    A.RandomBrightnessContrast(p=0.5),     # Variation luminosité/contraste (50%)
    A.GaussNoise(p=0.3)                    # Ajout de bruit (30%)
], additional_targets={'mask': 'mask'})    #  mêmes transformations au masque 


# Dossiers sources des images et masques REDIMENSIONNES
img_dir = "./Banques_images_rn_resized_V2/images"
mask_dir = "./Banques_images_rn_resized_V2/masks"

# Chemins des dossiers de destination (images et masques redimensionnés, augmentés)
aug_img_dir = "./Banques_images_aug_V2/images"
aug_mask_dir = "./Banques_images_aug_V2/masks"

# Création des dossiers de sortie
os.makedirs(aug_img_dir, exist_ok=True)
os.makedirs(aug_mask_dir, exist_ok=True)

# Nombre d'augmentations à générer par image originale
num_aug = 20

for fname in os.listdir(img_dir):
    #  On ignore ici les fichiers qui ne sont pas des images
    if not fname.lower().endswith((".jpg",".png")):
        continue

    # Charger image et masque
    img = np.array(Image.open(os.path.join(img_dir, fname)).convert("RGB"))
    mask_name = os.path.splitext(fname)[0] + "_mask.png"
    mask = np.array(Image.open(os.path.join(mask_dir, mask_name)))

    # On génére num_aug versions augmentées
    for i in range(num_aug):
        
        # Appliquer les transformations aléatoires
        augmented = transform(image=img, mask=mask)
        img_aug = augmented['image']
        mask_aug = augmented['mask']

        # Construire les noms de fichiers de sortie
        base = os.path.splitext(fname)[0]
        img_aug_name = f"{base}_aug{i}.png"
        mask_aug_name = f"{base}_aug{i}_mask.png"
        
        # Sauvegarder les versions augmentées
        Image.fromarray(img_aug).save(os.path.join(aug_img_dir, img_aug_name))
        Image.fromarray(mask_aug).save(os.path.join(aug_mask_dir, mask_aug_name))
        
print(f"\nImages augmentées sauvegardées dans : {aug_img_dir}")
print(f"Masques augmentés sauvegardés dans : {aug_mask_dir}\n")

#%%

# ════════════════════════════════════════════════════════════════════════════════
# ███████╗████████╗ █████╗ ██████╗ ███████╗    ██████╗ 
# ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝    ╚════██╗
# █████╗     ██║   ███████║██████╔╝█████╗       █████╔╝
# ██╔══╝     ██║   ██╔══██║██╔═══╝ ██╔══╝       ╚═══██╗
# ███████╗   ██║   ██║  ██║██║     ███████╗    ██████╔╝
# ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚══════╝    ╚═════╝ 
#                                                        
# ════════════════════════════════════════════════════════════════════════════════
#                   PRÉPARATION DU DATASET PYTORCH
# ════════════════════════════════════════════════════════════════════════════════
# Création d'une classe Dataset personnalisée pour charger les images et masques
# pendant l'entraînement.
# ════════════════════════════════════════════════════════════════════════════════

class FuseeDataset(Dataset):
    """
    Dataset personnalisé pour charger les images de fusée et leurs masques de segmentation.
    
    Args:
        img_dir (str): Chemin du dossier contenant les images
        mask_dir (str): Chemin du dossier contenant les masques
        image_list (list): Liste des noms de fichiers images à utiliser (train ou val)
    """
    
    def __init__(self, img_dir, mask_dir, image_list):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        
        # Liste des fichiers images du dossier (issue du split train / val)
        self.images = image_list

    def __len__(self):
        """Retourne le nombre total d'images dans le dataset"""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Charge et prépare une image et son masque pour l'entraînement.
        
        Args:
            idx (int): Index de l'image à charger
            
        Returns:
            tuple: (image_tensor, mask_tensor)
                - image_tensor: Tensor de forme (3, H, W) normalisé entre 0 et 1
                - mask_tensor: Tensor de forme (H, W) avec les labels de classe (0, 1, 2, 3)
        """
        # on récupéree les noms de fichiers
        img_name = self.images[idx]
        mask_name = os.path.splitext(img_name)[0] + "_mask.png"

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # ─────────────────────────────────────────────────────────────────────
        # Chargement et préparation de l'image
        # ─────────────────────────────────────────────────────────────────────
        
        # On ouvre l'image, et on la convertie en RGB
        image = Image.open(img_path).convert("RGB")
        
        # On convertie en numpy array et normaliser entre 0 et 1
        image = np.array(image, dtype=np.float32) / 255.0
        
        # On convertie en tensor PyTorch et réorganise les dimensions
        # De (H, W, C) vers (C, H, W)
        image = torch.tensor(image).permute(2, 0, 1)
        
        # ─────────────────────────────────────────────────────────────────────
        # Normalisation ImageNet (OBLIGATOIRE avec encodeur pré-entraîné)
        # ─────────────────────────────────────────────────────────────────────
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        image = (image - mean) / std
        
        # ─────────────────────────────────────────────────────────────────────
        # Chargement et préparation du masque
        # ─────────────────────────────────────────────────────────────────────
        
        # On ouvre les masques
        mask = Image.open(mask_path)
        
        # On convertie en numpy array (valeurs entières : 0, 1, 2, 3)
        mask = np.array(mask, dtype=np.int64)
        
        # Enfin on convertie en tensor PyTorch
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


all_images = [f for f in os.listdir("./Banques_images_aug_V2/images")
              if f.lower().endswith((".png", ".jpg"))]

train_imgs, val_imgs = train_test_split(
    all_images,
    test_size=0.2,
    random_state=42,
    shuffle=True
)


# ─────────────────────────────────────────────────────────────────────────────
# Création du dataset et du DataLoader
# ─────────────────────────────────────────────────────────────────────────────
train_dataset = FuseeDataset(
    "./Banques_images_aug_V2/images",
    "./Banques_images_aug_V2/masks",
    train_imgs
)

val_dataset = FuseeDataset(
    "./Banques_images_aug_V2/images",
    "./Banques_images_aug_V2/masks",
    val_imgs
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False)

print(f"Dataset créé : {len(train_dataset)} images d'entraînement")
print("Taille des batchs : 4 images\n")
#%%

# ════════════════════════════════════════════════════════════════════════════════
# ███████╗████████╗ █████╗ ██████╗ ███████╗    ██╗  ██╗
# ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝    ██║  ██║
# █████╗     ██║   ███████║██████╔╝█████╗      ███████║
# ██╔══╝     ██║   ██╔══██║██╔═══╝ ██╔══╝      ╚════██║
# ███████╗   ██║   ██║  ██║██║     ███████╗         ██║
# ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚══════╝         ╚═╝
#                                                        
# ════════════════════════════════════════════════════════════════════════════════
#                     CRÉATION DU MODÈLE U-NET
# ════════════════════════════════════════════════════════════════════════════════
# U-Net est une architecture très efficace pour la segmentation d'images.
# On utilise ResNet34 comme encodeur (pré-entraîné sur ImageNet).
# ════════════════════════════════════════════════════════════════════════════════

model = smp.Unet(
    encoder_name="resnet34",         # Encodeur : ResNet34
    encoder_weights="imagenet",      # Poids pré-entraînés sur ImageNet
    in_channels=3,                   # Entrée : image RGB (3 canaux)
    classes=4                        # Sortie : 4 classes (fond, fuselage, coiffe, ailerons), 
                                     # on peut aussi l'utiliser pour les mires, cest juste que 2 classes seront initulisées
)

# ─────────────────────────────────────────────────────────────────────────────
# Transfert sur GPU si disponible
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Device: {device}")

#%%

# ════════════════════════════════════════════════════════════════════════════════
# ███████╗████████╗ █████╗ ██████╗ ███████╗    ███████╗
# ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝    ██╔════╝
# █████╗     ██║   ███████║██████╔╝█████╗      ███████╗
# ██╔══╝     ██║   ██╔══██║██╔═══╝ ██╔══╝      ╚════██║
# ███████╗   ██║   ██║  ██║██║     ███████╗    ███████║
# ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚══════╝    ╚══════╝
#                                                        
# ════════════════════════════════════════════════════════════════════════════════
#                       ENTRAÎNEMENT DU MODÈLE
# ════════════════════════════════════════════════════════════════════════════════
# Configuration :
#   - Fonction de perte : CrossEntropyLoss (pour classification multi-classe)
#   - Optimiseur : Adam avec learning rate de 1e-4
#   - Métriques : Loss et accuracy (% de pixels correctement classés)
# ════════════════════════════════════════════════════════════════════════════════


def compute_metrics(outputs, masks, num_classes=4):
    """
    Calcule Dice (F1) et IoU pour un batch multiclasses.
    """
    preds = torch.argmax(outputs, dim=1)

    dice_total = 0.0
    iou_total  = 0.0

    # Calcul pour chaque classe sauf la classe "fond" (0)
    for cls in range(1, num_classes):
        pred_cls = (preds == cls).float()
        mask_cls = (masks == cls).float()

        intersection = (pred_cls * mask_cls).sum()
        union = pred_cls.sum() + mask_cls.sum()

        dice = (2.0 * intersection) / (union + 1e-7)
        dice_total += dice

        iou = intersection / (pred_cls.sum() + mask_cls.sum() - intersection + 1e-7)
        iou_total += iou

    # Moyenne sur les classes (hors fond)
    dice_mean = dice_total / (num_classes - 1)
    iou_mean  = iou_total / (num_classes - 1)

    return dice_mean.item(), iou_mean.item()




# =============================================================================
# CONFIG ENTRAÎNEMENT
# =============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 10
num_classes = 4

train_losses = []
val_losses   = []
val_dices    = []
val_ious     = []


# =============================================================================
# BOUCLE TRAIN + VALIDATION
# =============================================================================
for epoch in range(epochs):

    # -------------------------
    # TRAIN
    # -------------------------
    model.train()
    train_loss = 0.0

    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]"):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    val_loss = 0.0
    dice_scores = []
    iou_scores  = []

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VAL]"):
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            dice, iou = compute_metrics(outputs, masks)
            dice_scores.append(dice)
            iou_scores.append(iou)

    val_loss /= len(val_loader)
    mean_dice = float(np.mean(dice_scores))
    mean_iou  = float(np.mean(iou_scores))

    val_losses.append(val_loss)
    val_dices.append(mean_dice)
    val_ious.append(mean_iou)

    # -------------------------
    # LOG
    # -------------------------
    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Dice: {mean_dice:.4f} | "
        f"IoU: {mean_iou:.4f}"
    )


# =============================================================================
# SAUVEGARDE MODÈLE
# =============================================================================
torch.save(model.state_dict(), "modele_fusee_V2.pth")
print("\n✓ Modèle sauvegardé : modele_fusee_V2.pth")


# =============================================================================
# COURBES
# =============================================================================
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].plot(train_losses, label="Train Loss")
axs[0].plot(val_losses, label="Val Loss")
axs[0].set_title("Loss")
axs[0].legend()
axs[0].grid(alpha=0.3)

axs[1].plot(val_dices, label="Dice", color="green")
axs[1].set_title("Dice (Validation)")
axs[1].grid(alpha=0.3)

axs[2].plot(val_ious, label="IoU", color="orange")
axs[2].set_title("IoU (Validation)")
axs[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("courbes_segmentation.png", dpi=150)
plt.show()

print("✓ Courbes sauvegardées : courbes_segmentation.png")

#%%
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# ───────────────────────────────
# Charger le modèle
# ───────────────────────────────
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=4
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("modele_fusee_V3.pth", map_location=device))
model.to(device)
model.eval()

# ───────────────────────────────
# Charger l'image
# ───────────────────────────────
img_path = "./img9.jpg"  # ← change ici
img = Image.open(img_path).convert("RGB").resize((512, 512))
img_tensor = torch.tensor(np.array(img, dtype=np.float32)/255.0).permute(2,0,1).unsqueeze(0)

# Normalisation ImageNet
mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
img_tensor = (img_tensor - mean) / std
img_tensor = img_tensor.to(device)

# ───────────────────────────────
# Prédiction
# ───────────────────────────────
with torch.no_grad():
    output = model(img_tensor)
    pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# ───────────────────────────────
# Affichage
# ───────────────────────────────
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Image originale")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(pred_mask, cmap="jet", vmin=0, vmax=3)
plt.title("Masque prédit")
plt.axis("off")
plt.show()

