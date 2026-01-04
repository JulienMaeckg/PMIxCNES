from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.uix.image import Image as KivyImage
from kivy.uix.progressbar import ProgressBar
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.utils import platform
from kivy.clock import Clock
from kivy.metrics import sp
from kivy.app import App

from urllib.parse import urlparse, parse_qs
from PIL import Image
import threading
import requests
import tempfile
import time
import gc
import os

# Importation des modules locaux
import shapeExtraction
import noseConeShape
import spockApi

# Bloc conditionnel pour les fonctionnalités spécifiques au système Android
if platform == "android":
    # Importation des outils de demande de permissions système
    from android.permissions import request_permissions, Permission

    # Importation de la référence à l'activité Android courante
    from android import activity

    # Importation de l'outil de pont entre Python et les classes Java
    from jnius import autoclass

    # Importation de l'outil de sélection de fichiers natif
    from plyer import filechooser

    # Récupération des classes Java nécessaires via Pyjnius
    Intent = autoclass("android.content.Intent")
    PythonActivity = autoclass("org.kivy.android.PythonActivity")
    MediaStore = autoclass("android.provider.MediaStore")
    Uri = autoclass("android.net.Uri")
    File = autoclass("java.io.File")
    FileProvider = autoclass("androidx.core.content.FileProvider")
    Environment = autoclass("android.os.Environment")
    FileOutputStream = autoclass("java.io.FileOutputStream")
    CompressFormat = autoclass("android.graphics.Bitmap$CompressFormat")
    Bitmap = autoclass("android.graphics.Bitmap")
else:
    # Initialisation de la variable activity à None sur les systèmes non-Android
    activity = None


def get_android_launch_url():
    """Récupère l'URL ayant servi à lancer l'application sur Android (Deep Linking)."""
    try:
        # Importation locale pour éviter les erreurs sur Desktop
        from jnius import autoclass

        # Accès à l'instance de l'activité Kivy en cours
        PythonActivity = autoclass("org.kivy.android.PythonActivity")
        activity = PythonActivity.mActivity

        # Récupération de l'Intent qui a démarré l'activité
        intent = activity.getIntent()
        # Extraction de l'action associée à l'Intent
        action = intent.getAction()

        # Vérification si l'action est une demande de visualisation de données
        if action == "android.intent.action.VIEW":
            # Extraction des données (URI) de l'Intent
            uri = intent.getData()
            # Conversion de l'URI Java en chaîne de caractères Python
            if uri:
                return uri.toString()
    except Exception as e:
        # Affichage de l'erreur en cas d'échec de récupération de l'Intent
        print("Erreur Intent:", e)

    # Retourne None si aucune URL valide n'est trouvée
    return None


def on_new_intent(intent):
    """Gère les nouveaux Intents reçus alors que l'application est déjà en mémoire."""
    try:
        # Extraction de l'action de l'Intent reçu
        action = intent.getAction()
        # Vérification s'il s'agit d'une URL de type VIEW
        if action == "android.intent.action.VIEW":
            # Récupération de l'URI
            uri = intent.getData()
            # Conversion de l'URI en chaîne
            if uri:
                return uri.toString()
    except Exception as e:
        # Journalisation de l'erreur d'analyse du nouvel Intent
        print("Erreur nouveau Intent:", e)
    # Retourne None par défaut
    return None


def open_website(url):
    """Ouvre un navigateur web externe vers l'URL spécifiée sur Android."""
    try:
        # Importation des classes Java nécessaires pour l'action VIEW
        from jnius import autoclass, cast

        Intent = autoclass("android.content.Intent")
        Uri = autoclass("android.net.Uri")
        PythonActivity = autoclass("org.kivy.android.PythonActivity")

        # Création d'un Intent avec l'action de visualisation standard
        intent = Intent(Intent.ACTION_VIEW)
        # Définition de la cible de l'Intent avec l'URL parsée
        intent.setData(Uri.parse(url))

        # Récupération de l'activité courante pour lancer l'Intent
        current_activity = PythonActivity.mActivity
        current_activity.startActivity(intent)
    except Exception as e:
        # Affichage de l'erreur en cas d'échec d'ouverture du navigateur
        print("Erreur ouverture site :", e)


def get_code_from_SPOCK(id_, key):
    """Interroge l'API SPOCK pour obtenir un code de session à partir des identifiants."""
    try:
        # Construction de l'URL de l'API avec les paramètres fournis
        url = f"https://www.planete-sciences.org/espace/SPOCK/api.html?object=stabtraj&id={id_}&key={key}"
        # Envoi d'une requête GET HTTP
        r = requests.get(url)
        # Vérification du succès de la requête
        if r.status_code == 200:
            # Extraction du contenu JSON
            data = r.json()
            # Récupération de la valeur associée à la clé "code"
            code = data.get("code", "")
            return code
        else:
            # Affichage de l'erreur HTTP si le serveur répond négativement
            print(f"Erreur récupération code: {r.status_code}")
            return None
    except Exception as e:
        # Gestion des erreurs de connexion ou d'analyse JSON
        print(f"Erreur request code: {e}")
        return None


class RoundedButton(Button):
    """Classe personnalisée créant un bouton avec des coins arrondis et un fond coloré."""

    def __init__(self, **kwargs):
        # Récupération de la couleur de fond personnalisée, gris par défaut
        self._bg_color = kwargs.pop("background_color", (0.8, 0.8, 0.8, 1))

        # Initialisation de la classe parente Button
        super().__init__(**kwargs)

        # Désactivation du rendu visuel par défaut du bouton Kivy
        self.background_color = (0, 0, 0, 0)
        self.background_normal = ""
        self.background_down = ""

        # Nettoyage des instructions graphiques précédentes
        self.canvas.before.clear()
        # Définition du nouveau style graphique avant le rendu du texte
        with self.canvas.before:
            # Définition de la couleur de dessin
            self.bg_color = Color(rgba=self._bg_color)
            # Création du rectangle arrondi calqué sur la taille du widget
            self.bg_rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[25])

        # Liaison des changements de position/taille à la mise à jour du dessin
        self.bind(pos=self._update_rect, size=self._update_rect)
        # Liaison des événements presser/relâcher pour l'effet visuel
        self.bind(on_press=self._on_press, on_release=self._on_release)

    def _update_rect(self, *args):
        """Met à jour la position et la taille du rectangle arrondi lors du redimensionnement."""
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size

    def _on_press(self, *args):
        """Change la couleur de fond en gris foncé lors de l'appui sur le bouton."""
        self.bg_color.rgba = (0.4, 0.4, 0.4, 1)

    def _on_release(self, *args):
        """Restaure la couleur de fond initiale lorsque le bouton est relâché."""
        self.bg_color.rgba = self._bg_color


class RoundedSpinner(Spinner):
    """Classe personnalisée pour un menu déroulant aux coins arrondis."""

    def __init__(self, **kwargs):
        # Extraction de la couleur de fond
        self._bg_color = kwargs.pop("background_color", (0.8, 0.8, 0.8, 1))

        # Initialisation de la classe parente Spinner
        super().__init__(**kwargs)

        # Mise en transparence des textures par défaut
        self.background_color = [0, 0, 0, 0]
        self.background_normal = ""
        self.background_down = ""

        # Dessin du fond arrondi personnalisé
        self.canvas.before.clear()
        with self.canvas.before:
            self.bg_color = Color(*self._bg_color)
            self.bg_rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[25])

        # Liaison pour la mise à jour dynamique du rectangle
        self.bind(pos=self._update_rect, size=self._update_rect)

        # Appel de la configuration esthétique de la liste déroulante
        self._setup_dropdown()

    def _update_rect(self, *args):
        """Ajuste le rectangle graphique à la taille du Spinner."""
        if hasattr(self, "bg_rect"):
            self.bg_rect.pos = self.pos
            self.bg_rect.size = self.size

    def _setup_dropdown(self):
        """Définit le style visuel des options apparaissant dans le menu déroulant."""

        def customize_dropdown(*args):
            """Applique les styles aux boutons de la liste déroulante."""
            if hasattr(self, "_dropdown") and self._dropdown:
                dropdown = self._dropdown

                # Configuration du conteneur de la liste (fond blanc)
                dropdown.background_color = (1, 1, 1, 1)
                dropdown.spacing = sp(1)

                # Nettoyage des options existantes pour les reconstruire proprement
                dropdown.clear_widgets()

                for value in self.values:
                    # Calcul de la taille de police adaptée
                    dropdown_font_size = max(self.font_size, sp(20))

                    # Création d'un bouton pour chaque valeur du Spinner
                    btn = Button(
                        text=value,
                        size_hint_y=None,
                        height=sp(60),
                        background_normal="",
                        background_down="",
                        background_color=(0.95, 0.95, 0.95, 1),
                        color=(0, 0, 0, 1),
                        font_size=dropdown_font_size,
                    )

                    # Définition des fonctions pour l'effet de survol/appui
                    def on_btn_press(instance):
                        instance.background_color = (0.85, 0.85, 0.85, 1)

                    def on_btn_release(instance):
                        instance.background_color = (0.95, 0.95, 0.95, 1)

                    # Attribution des effets visuels au bouton
                    btn.bind(on_press=on_btn_press)
                    btn.bind(on_release=on_btn_release)

                    # Liaison du clic sur l'option pour sélectionner la valeur et fermer la liste
                    btn.bind(
                        on_release=lambda btn_instance: dropdown.select(
                            btn_instance.text
                        )
                    )
                    # Ajout de l'option au widget de liste
                    dropdown.add_widget(btn)

        # Planification de la personnalisation après l'initialisation du widget
        Clock.schedule_once(customize_dropdown, 0.1)

        # Ré-application du style si la liste des valeurs est modifiée dynamiquement
        self.bind(values=customize_dropdown)


class VulcainApp(App):
    """Classe principale de l'application Vulcain."""

    def build(self):
        # Gestion des permissions au lancement sur Android
        if platform == "android":
            request_permissions(
                [
                    Permission.CAMERA,
                    Permission.READ_EXTERNAL_STORAGE,
                    Permission.WRITE_EXTERNAL_STORAGE,
                    Permission.READ_MEDIA_IMAGES,
                    Permission.INTERNET,
                ]
            )

        # Initialisation des variables de stockage des identifiants SPOCK
        self.SPOCK_id = ""
        self.SPOCK_key = ""
        self.SPOCK_code = ""
        # Mémoire des derniers identifiants pour éviter les doublons de traitement
        self.derniers_identifiants_recus = {"id": "", "key": "", "code": ""}
        # Flags de contrôle pour la gestion des Intents asynchrones
        self.intent_en_cours_de_traitement = False
        self.dernier_intent_traite = None
        # Références pour le suivi du calcul de stabilité
        self.popup_progression = None
        self.calcul_termine_flag = None
        self.temps_debut_calcul = None
        # Variables d'état de l'affichage utilisateur
        self.texte_base = "EN ATTENTE"
        self.couleur = "000000"
        self.temps = 15.0  # Durée estimée du calcul pour la barre de progression

        # Définition d'une largeur cible pour l'adaptation de l'interface
        self.reference_width = 550

        # Calcul initial du facteur d'échelle des widgets
        self.update_scale_factor()

        # Surveillance du redimensionnement de la fenêtre pour l'adaptabilité
        Window.bind(on_resize=self.on_window_resize)

        # Mise en place du fond blanc de l'application
        with Window.canvas.before:
            Color(1, 1, 1, 1)
            self.bg_rect = Rectangle(pos=(0, 0), size=Window.size)
        # Liaison de la taille du fond à celle de la fenêtre
        Window.bind(size=lambda instance, value: setattr(self.bg_rect, "size", value))

        # Initialisation des variables liées à la capture d'image
        self.chemin_image_fusee = None
        self.photo_uri = None
        # Codes de requête pour l'identification des résultats d'activités Android
        self.REQUEST_IMAGE_CAPTURE = 1
        self.REQUEST_CSV_LOAD = 2

        # Initialisation de l'état d'avancement du scan et du calcul
        self.fusee_scannee = 0
        self.fusee_dimensions = 0
        self.afficher_det = 0
        # Dictionnaire global contenant les mesures extraites de la fusée
        self.donnees_completes = {}
        self.donnees_completes["Q_ail"] = None
        self.donnees_completes["Q_can"] = None

        # Liaison des événements Android (résultats d'activités et nouveaux liens)
        if platform == "android":
            activity.bind(on_activity_result=self.on_activity_result)
            activity.bind(on_new_intent=self.on_new_intent_received)

        # Construction de l'interface utilisateur principale (Layout vertical)
        layout = BoxLayout(
            orientation="vertical", padding=self.pad(10), spacing=self.pad(10)
        )

        # Titre de l'application
        self.title_label = Label(
            text="VULCAIN",
            font_size=self.sp(45),
            size_hint_y=0.1,
            color=(0, 0, 0, 1),
            bold=True,
        )
        layout.add_widget(self.title_label)

        # Étiquette de section pour les ailerons
        self.ailerons_label = Label(
            text="Ailerons", font_size=self.sp(35), size_hint_y=0.1, color=(0, 0, 0, 1)
        )
        # layout.add_widget(self.ailerons_label)

        # Ligne de configuration du nombre d'ailerons inférieurs
        aileron_bas_layout = BoxLayout(orientation="horizontal", size_hint_y=0.1)
        self.aileron_bas_label = Label(
            text="Nombre d'ailerons :",
            font_size=self.sp(25),
            color=(0, 0, 0, 1),
            size_hint_x=0.6,
        )
        aileron_bas_layout.add_widget(self.aileron_bas_label)
        # Menu déroulant pour le choix du nombre d'ailerons (3 à 6)
        self.aileron_bas_spinner = RoundedSpinner(
            text="4",
            values=("3", "4", "5", "6"),
            size_hint_x=0.4,
            font_size=self.sp(25),
            background_color=(0.8, 0.8, 0.8, 1),
            color=(0, 0, 0, 1),
        )
        # Liaison pour la mise à jour de la police à l'ouverture du menu
        self.aileron_bas_spinner.bind(on_release=self._on_spinner_open)
        # Liaison pour réinitialiser l'état si l'utilisateur change de configuration
        self.aileron_bas_spinner.bind(text=self.on_aileron_change)
        aileron_bas_layout.add_widget(self.aileron_bas_spinner)
        layout.add_widget(aileron_bas_layout)

        # Section pour les ailerons supérieurs (canards) - Configuration similaire
        aileron_haut_layout = BoxLayout(orientation="horizontal", size_hint_y=0.1)
        self.aileron_haut_label = Label(
            text="Ailerons du haut :",
            font_size=self.sp(25),
            color=(0, 0, 0, 1),
            size_hint_x=0.6,
        )
        aileron_haut_layout.add_widget(self.aileron_haut_label)
        self.aileron_haut_spinner = RoundedSpinner(
            text="0",
            values=("0", "2", "3", "4", "5", "6"),
            size_hint_x=0.4,
            font_size=self.sp(25),
            background_color=(0.8, 0.8, 0.8, 1),
            color=(0, 0, 0, 1),
        )
        self.aileron_haut_spinner.bind(on_release=self._on_spinner_open)
        self.aileron_haut_spinner.bind(text=self.on_aileron_change)
        aileron_haut_layout.add_widget(self.aileron_haut_spinner)
        # layout.add_widget(aileron_haut_layout)

        # Étiquette de section pour le scannage
        self.scannage_label = Label(
            text="Scannage", font_size=self.sp(35), size_hint_y=0.1, color=(0, 0, 0, 1)
        )
        layout.add_widget(self.scannage_label)

        # Boutons d'action pour le scan et le calcul
        scannage_buttons_layout = BoxLayout(
            orientation="horizontal", size_hint_y=0.1, spacing=self.pad(10)
        )
        self.bouton_scan = RoundedButton(
            text="Scanner fusée",
            background_color=(0.8, 0.8, 0.8, 1),
            font_size=self.sp(25),
            color=(0, 0, 0, 1),
        )
        self.bouton_scan.bind(on_release=self.afficher_choix_source_image)
        self.bouton_calcul = RoundedButton(
            text="Extraire dimensions",
            background_color=(0.8, 0.8, 0.8, 1),
            font_size=self.sp(25),
            color=(0, 0, 0, 1),
        )
        self.bouton_calcul.bind(on_release=self.lancer_calcul_stabilite)
        scannage_buttons_layout.add_widget(self.bouton_scan)
        scannage_buttons_layout.add_widget(self.bouton_calcul)
        layout.add_widget(scannage_buttons_layout)

        # Label de statut affichant l'état courant de l'application (EN ATTENTE, etc.)
        self.infos_label = Label(
            text="[b]EN ATTENTE[/b]",
            markup=True,
            font_size=self.sp(30),
            size_hint_y=0.1,
            color=(0, 0, 0, 1),
        )
        layout.add_widget(self.infos_label)

        # Zone des boutons de gestion des résultats et remise à zéro
        texte_buttons_layout = BoxLayout(
            orientation="vertical", size_hint_y=0.2, spacing=self.pad(10)
        )
        self.bouton_details = RoundedButton(
            text="Afficher les résultats",
            background_color=(0.8, 0.8, 0.8, 1),
            font_size=self.sp(25),
            color=(0, 0, 0, 1),
            size_hint_y=0.5,
        )
        # TODO: Ajouter la fonction self.afficher_details au bind du bouton_details
        self.bouton_reset = RoundedButton(
            text="Réinitialiser",
            background_color=(0.8, 0.8, 0.8, 1),
            color=(0, 0, 0, 1),
            font_size=self.sp(25),
            size_hint_y=0.5,
        )
        self.bouton_reset.bind(on_release=self.reset_donnees)
        self.bouton_details.bind(on_release=self.afficher_details)
        texte_buttons_layout.add_widget(self.bouton_details)
        texte_buttons_layout.add_widget(self.bouton_reset)
        layout.add_widget(texte_buttons_layout)

        # Zone des boutons utilitaires en bas d'écran
        bottom_buttons_layout = BoxLayout(
            orientation="vertical", size_hint_y=0.3, spacing=self.pad(10)
        )
        self.bouton_identifiants = RoundedButton(
            text="Identification SPOCK",
            background_color=(0.8, 0.8, 0.8, 1),
            font_size=self.sp(25),
            color=(0, 0, 0, 1),
        )
        self.bouton_identifiants.bind(on_release=self.afficher_identifiants_SPOCK)
        self.bouton_send_api = RoundedButton(
            text="Envoyer les données",
            background_color=(0.8, 0.8, 0.8, 1),
            font_size=self.sp(25),
            color=(0, 0, 0, 1),
        )
        self.bouton_guide = RoundedButton(
            text="Guide d'utilisation",
            background_color=(0.8, 0.8, 0.8, 1),
            font_size=self.sp(25),
            color=(0, 0, 0, 1),
        )
        self.bouton_guide.bind(on_release=self.afficher_guide)
        bottom_buttons_layout.add_widget(self.bouton_identifiants)
        bottom_buttons_layout.add_widget(self.bouton_send_api)
        self.bouton_send_api.bind(on_release=self.send_api)
        bottom_buttons_layout.add_widget(self.bouton_guide)
        layout.add_widget(bottom_buttons_layout)

        # Retourne le layout complet pour l'affichage
        return layout

    def on_start(self):
        """Action exécutée immédiatement après le démarrage de l'application."""
        if platform == "android":
            # Planifie la lecture de l'URL de lancement après 1 seconde
            Clock.schedule_once(self.read_SPOCK_link, 1)

    def on_resume(self):
        """Action exécutée quand l'application revient au premier plan."""
        if platform == "android":
            # Vérification de nouveaux Intents si aucun n'est déjà en cours de traitement
            if not self.intent_en_cours_de_traitement:
                Clock.schedule_once(self.check_for_new_intent, 0.3)

    def on_new_intent_received(self, intent):
        """Gère l'arrivée d'un nouvel Intent (Deep Link) pendant l'activité."""
        print("[INTENT] Nouvel intent reçu")
        if not self.intent_en_cours_de_traitement:
            # Analyse de l'Intent via la fonction globale
            link = on_new_intent(intent)

            # Évite de traiter deux fois exactement le même lien
            if link == self.dernier_intent_traite:
                print("[INTENT] Intent déjà traité, ignoré")
                return

            # Valide s'il s'agit d'un lien SPOCK et lance le traitement
            if link and self._is_spock_link(link):
                self.intent_en_cours_de_traitement = True
                self.dernier_intent_traite = link
                Clock.schedule_once(lambda dt: self._traiter_intent_spock(link), 0.1)

    def check_for_new_intent(self, dt):
        """Vérifie manuellement la présence d'un Intent non traité."""
        if self.intent_en_cours_de_traitement:
            print("[INTENT] Traitement d'intent déjà en cours, ignoré")
            return

        try:
            # Récupération de l'instance Java de l'activité
            from jnius import autoclass

            PythonActivity = autoclass("org.kivy.android.PythonActivity")
            activity_instance = PythonActivity.mActivity
            intent = activity_instance.getIntent()

            # Analyse de l'Intent courant
            link = on_new_intent(intent)

            # Protection contre les traitements redondants
            if link == self.dernier_intent_traite:
                print("[INTENT] Intent déjà traité, ignoré")
                return

            # Lancement du traitement si le lien est conforme
            if link and self._is_spock_link(link):
                self.intent_en_cours_de_traitement = True
                self.dernier_intent_traite = link
                self._traiter_intent_spock(link)
        except Exception as e:
            # Journalisation des erreurs d'accès à l'API Android
            print("[INTENT] Erreur vérification intent:", e)

    def _is_spock_link(self, link):
        """Valide si une chaîne est un deep link Vulcain/SPOCK conforme."""
        if not link:
            return False

        try:
            # Rejet des liens systèmes Android qui ne sont pas des URLs web/deep-links
            if link.startswith("content://") or link.startswith("file://"):
                print(f"[INTENT] URI de contenu Android ignoré: {link[:50]}...")
                return False

            # Décomposition de l'URL
            parsed = urlparse(link)

            print(f"[INTENT] Analyse du lien: {link}")
            print(f"[INTENT] Scheme: {parsed.scheme}")
            print(f"[INTENT] Host: {parsed.netloc}")

            # Validation du protocole personnalisé
            if parsed.scheme != "vulcainpmi":
                print(
                    f"[INTENT] Scheme incorrect (attendu: vulcainpmi, reçu: {parsed.scheme})"
                )
                return False

            # Validation de l'hôte
            if parsed.netloc != "ouvrir":
                print(
                    f"[INTENT] Host incorrect (attendu: ouvrir, reçu: {parsed.netloc})"
                )
                return False

            # Vérification de la présence d'au moins un paramètre d'identification
            params = parse_qs(parsed.query)
            has_spock_params = any(param in params for param in ["id", "key", "code"])

            print(f"[INTENT] Paramètres trouvés: {list(params.keys())}")
            print(f"[INTENT] A des paramètres SPOCK: {has_spock_params}")

            return has_spock_params

        except Exception as e:
            # Capture des erreurs d'analyse syntaxique de l'URL
            print(f"[INTENT] Erreur validation lien: {e}")
            return False

    def process_SPOCK_link(self, link):
        """Extrait les identifiants d'un lien validé et met à jour l'application."""

        # Extraction des paramètres de la query string
        parsed = urlparse(link)
        params = parse_qs(parsed.query)

        # Récupération sécurisée des valeurs (prend la première occurrence ou vide)
        new_id = params.get("id", [""])[0]
        new_key = params.get("key", [""])[0]
        new_code = params.get("code", [""])[0]

        print(
            f"[SPOCK] Traitement du lien - ID: {new_id}, Key: {new_key}, Code: {new_code}"
        )

        if new_id or new_key or new_code:
            # Détection de changements par rapport aux données en mémoire
            id_changed = (
                new_id != "" and new_id != self.derniers_identifiants_recus["id"]
            )
            key_changed = (
                new_key != "" and new_key != self.derniers_identifiants_recus["key"]
            )
            code_changed = (
                new_code != "" and new_code != self.derniers_identifiants_recus["code"]
            )

            # Mise à jour des variables si des données fraîches sont présentes
            if id_changed or key_changed or code_changed:
                print(f"[SPOCK] Changement détecté - Affichage du message")

                if new_id:
                    self.SPOCK_id = new_id
                    self.derniers_identifiants_recus["id"] = new_id
                if new_key:
                    self.SPOCK_key = new_key
                    self.derniers_identifiants_recus["key"] = new_key
                if new_code:
                    self.SPOCK_code = new_code
                    self.derniers_identifiants_recus["code"] = new_code

            # Notification visuelle de la réception des données
            self.infos_label.text = "[b][color=0000ff]IDENTIFIANTS REÇUS ![/color][/b]"

            # Planification du retour à l'état d'affichage normal après 2.5s
            Clock.schedule_once(
                lambda dt: setattr(
                    self.infos_label,
                    "text",
                    f"[b][color={self.couleur}]{self.texte_base}[/color][/b]",
                ),
                2.5,
            )

            print(
                f"[SPOCK] Identifiants actuels - ID: {self.SPOCK_id}, Key: {self.SPOCK_key}, Code: {self.SPOCK_code}"
            )

    def _traiter_intent_spock(self, link):
        """Encapsule le traitement d'un lien pour gérer le verrouillage d'état."""
        try:
            # Appel du traitement effectif
            self.process_SPOCK_link(link)
        finally:
            # Libération du flag après un court délai pour éviter les rebonds
            Clock.schedule_once(
                lambda dt: setattr(self, "intent_en_cours_de_traitement", False), 0.5
            )

    def read_SPOCK_link(self, dt):
        """Récupère et traite le lien de démarrage au chargement."""
        if self.intent_en_cours_de_traitement:
            return

        # Récupération de l'URL via l'API Android
        link = get_android_launch_url()

        # Vérification de doublon
        if link == self.dernier_intent_traite:
            print("[INTENT] Intent de lancement déjà traité, ignoré")
            return

        # Lancement du traitement si valide
        if link and self._is_spock_link(link):
            self.intent_en_cours_de_traitement = True
            self.dernier_intent_traite = link
            self._traiter_intent_spock(link)

    def afficher_message_temporaire(self, texte, couleur, delai=0.6):
        """Affiche une animation d'attente puis un message coloré final."""
        # Affichage immédiat du statut "En cours"
        self.infos_label.text = "[i][color=000000]En cours...[/color][/i]"

        # Planification de l'affichage du message définitif après le délai
        Clock.schedule_once(
            lambda dt: setattr(
                self.infos_label, "text", f"[b][color={couleur}]{texte}[/color][/b]"
            ),
            delai,
        )

    def update_scale_factor(self):
        """Calcule le ratio entre la largeur réelle de l'écran et la largeur de référence."""
        self.scale_factor = Window.width / self.reference_width

    def pad(self, size):
        """Adapte une valeur de marge/remplissage au facteur d'échelle de l'écran."""
        return int(size * self.scale_factor)

    def sp(self, size):
        """Adapte une taille de police au facteur d'échelle de l'écran."""
        return int(size * self.scale_factor)

    def on_window_resize(self, window, width, height):
        """Réactualise l'échelle et les polices lorsque la fenêtre change de taille."""
        self.update_scale_factor()
        self.update_all_font_sizes()

    def update_all_font_sizes(self):
        """Applique récursivement les nouvelles tailles de police à tous les widgets clés."""
        # Liste des tuples (référence du widget, taille de police initiale)
        widgets_to_update = [
            (self.title_label, 45),
            (self.ailerons_label, 35),
            (self.aileron_bas_label, 25),
            (self.aileron_bas_spinner, 25),
            (self.aileron_haut_label, 25),
            (self.aileron_haut_spinner, 25),
            (self.scannage_label, 35),
            (self.bouton_scan, 25),
            (self.bouton_calcul, 25),
            (self.infos_label, 30),
            (self.bouton_details, 25),
            (self.bouton_reset, 25),
            (self.bouton_send_api, 25),
            (self.bouton_identifiants, 25),
            (self.bouton_guide, 25),
        ]
        # Boucle de mise à jour utilisant la méthode sp()
        for widget, size in widgets_to_update:
            widget.font_size = self.sp(size)

    def _on_spinner_open(self, spinner):
        """Déclenche la mise à jour des polices des options dès l'ouverture du menu déroulant."""
        Clock.schedule_once(lambda dt: self._update_spinner_font(spinner), 0.025)

    def _update_spinner_font(self, spinner):
        """Parcourt les éléments du menu déroulant pour ajuster leur taille de texte."""
        dropdown = getattr(spinner, "_dropdown", None)
        if dropdown and hasattr(dropdown, "container"):
            # Application de la taille mise à l'échelle sur chaque bouton enfant de la liste
            for child in dropdown.container.children:
                if hasattr(child, "font_size"):
                    child.font_size = self.sp(25)

    def reset_donnees(self, instance):
        """Réinitialise toutes les variables de calcul et l'état visuel de l'app."""
        # Remise à zéro des mesures
        self.donnees_completes = {}
        self.donnees_completes["Q_ail"] = None
        self.donnees_completes["Q_can"] = None
        # Remise à zéro des menus de sélection
        self.aileron_bas_spinner.text = "4"
        self.aileron_haut_spinner.text = "0"
        # Reset des couleurs et textes de base
        couleur = "000000"
        texte = "EN ATTENTE"
        self.texte_base = "EN ATTENTE"
        self.couleur = "000000"
        # Notification visuelle de la réinitialisation
        self.afficher_message_temporaire(texte, couleur)
        # Reset des drapeaux de progression
        self.fusee_scannee = 0
        self.fusee_dimensions = 0
        self.afficher_det = 0
        # Effacement des identifiants
        self.SPOCK_key = ""
        self.SPOCK_id = ""
        self.SPOCK_code = ""
        self.derniers_identifiants_recus = {"id": "", "key": "", "code": ""}

        # Suppression sécurisée du fichier image temporaire stocké sur l'appareil
        if self.chemin_image_fusee and os.path.exists(self.chemin_image_fusee):
            try:
                if platform == "android":
                    os.remove(self.chemin_image_fusee)
                print(f"[RESET] Image temporaire supprimée: {self.chemin_image_fusee}")
            except Exception as e:
                print(f"[RESET] Erreur suppression image: {e}")

        # Nettoyage de la référence et appel au ramasse-miettes
        self.chemin_image_fusee = None
        gc.collect()

    def on_aileron_change(self, spinner, text):
        """Réagit au changement de nombre d'ailerons pour forcer une nouvelle validation."""
        # Si une extraction avait déjà été faite
        if self.afficher_det == 1 and self.fusee_dimensions == 1:
            # Désactivation du flag d'affichage pour obliger à revoir les détails
            self.afficher_det = 0

            # Repassage au statut "Données extraites" (vert)
            couleur = "00ff00"
            texte = "DONNÉES EXTRAITES !"
            self.texte_base = texte
            self.couleur = couleur

            # Mise à jour immédiate du texte d'information
            self.infos_label.text = f"[b][color={couleur}]{texte}[/color][/b]"

    def afficher_choix_source_image(self, instance):
        """Crée un popup permettant à l'utilisateur de choisir entre Appareil Photo ou Galerie."""
        if platform == "android":
            # Création du conteneur vertical pour les boutons de choix
            content = BoxLayout(
                orientation="vertical", spacing=self.pad(10), padding=self.pad(10)
            )

            # Bouton pour déclencher la caméra
            btn_camera = RoundedButton(
                text="Prendre une photo",
                background_color=(0.8, 0.8, 0.8, 1),
                font_size=self.sp(27),
                color=(0, 0, 0, 1),
                size_hint_y=0.4,
            )

            # Bouton pour ouvrir la galerie d'images
            btn_galerie = RoundedButton(
                text="Choisir depuis la galerie",
                background_color=(0.8, 0.8, 0.8, 1),
                font_size=self.sp(27),
                color=(0, 0, 0, 1),
                size_hint_y=0.4,
            )

            # Bouton de fermeture du popup
            btn_fermer = RoundedButton(
                text="Fermer",
                size_hint_y=0.2,
                background_color=(0.8, 0.8, 0.8, 1),
                font_size=self.sp(25),
                color=(0, 0, 0, 1),
            )

            # Ajout des boutons au layout
            content.add_widget(btn_camera)
            content.add_widget(btn_galerie)
            content.add_widget(btn_fermer)

            # Création et configuration de la fenêtre modale
            popup = Popup(
                title="Choisir une source",
                content=content,
                size_hint=(0.9, 0.5),
                title_size=self.sp(25),
            )

            # Définition des actions internes au popup
            def prendre_photo(btn_instance):
                popup.dismiss()
                self._ouvrir_camera_android()

            def choisir_galerie(btn_instance):
                popup.dismiss()
                self._ouvrir_galerie_android()

            # Liaison des actions aux boutons respectifs
            btn_camera.bind(on_release=prendre_photo)
            btn_galerie.bind(on_release=choisir_galerie)
            btn_fermer.bind(on_release=popup.dismiss)

            # Affichage du popup
            popup.open()
        else:
            # Comportement de simulation pour le développement sur ordinateur (Desktop)
            self.chemin_image_fusee = "fusee_test.jpg"
            couleur = "00ff00"
            texte = "FUSÉE SCANNÉE !"
            self.fusee_scannee = 1
            self.texte_base = texte
            self.couleur = couleur
            self.afficher_message_temporaire(texte, couleur)

    def _ouvrir_camera_android(self):
        """Invoque l'application caméra par défaut d'Android via un Intent."""
        if platform == "android":
            try:
                # Récupération de l'activité
                current_activity = PythonActivity.mActivity
                # Création de l'Intent de capture d'image
                intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                # Lancement de l'appareil photo en attendant un résultat
                current_activity.startActivityForResult(
                    intent, self.REQUEST_IMAGE_CAPTURE
                )
                print("[CAMERA] Camera intent started")
            except Exception as e:
                # Gestion des erreurs d'accès matériel
                print(f"[CAMERA] ERREUR: {e}")
                couleur = "ff0000"
                texte = "ERREUR CAMÉRA..."
                self.texte_base = texte
                self.couleur = couleur
                self.afficher_message_temporaire(texte, couleur)

    def _ouvrir_galerie_android(self):
        """Ouvre le sélecteur de fichiers du système pour choisir une image."""
        if platform == "android":
            try:
                # Utilisation de la bibliothèque Plyer pour l'abstraction du sélecteur
                filechooser.open_file(
                    on_selection=self._on_image_selected_from_galery,
                    filters=["*.jpg", "*.png", "*.jpeg"],
                )
            except Exception as e:
                # Notification en cas d'échec d'ouverture de la galerie
                print(f"GALERIE ERREUR: {e}")
                couleur = "ff0000"
                texte = "ERREUR GALERIE..."
                self.texte_base = texte
                self.couleur = couleur
                self.afficher_message_temporaire(texte, couleur)

    def _on_image_selected_from_galery(self, file_paths):
        """Reçoit et traite le chemin du fichier sélectionné dans la galerie."""
        if file_paths:
            # Envoi du premier chemin reçu à la validation
            self._traiter_image_valider_chemin(file_paths[0])
        else:
            # Cas où l'utilisateur a annulé la sélection
            couleur = "ff0000"
            texte = "AUCUNE IMAGE SÉLECTIONNÉE..."
            self.texte_base = texte
            self.couleur = couleur
            self.afficher_message_temporaire(texte, couleur)

    def on_activity_result(self, request_code, result_code, data):
        """Intercepte le retour d'une application tierce (Caméra/Galerie)."""
        RESULT_OK = -1  # Valeur standard Android pour un succès

        print(f"[ACTIVITY] Resultat - requete: {request_code}, resultat: {result_code}")

        # Vérification si l'action a été annulée ou a échoué
        if result_code != RESULT_OK:
            couleur = "000000"
            texte = "EN ATTENTE"
            self.texte_base = texte
            self.couleur = couleur
            self.afficher_message_temporaire(texte, couleur)
            return

        # Identification de l'origine du résultat (Appareil photo)
        if request_code == self.REQUEST_IMAGE_CAPTURE:
            print("[CAMERA] Traitement du resultat camera...")
            # Traitement différé sur le thread principal pour éviter les gels d'UI
            Clock.schedule_once(lambda dt: self._traiter_resultat_camera(data), 0)

        # Gestion optionnelle d'un fichier CSV (prévu mais non implémenté)
        elif request_code == self.REQUEST_CSV_LOAD:
            pass

    def _traiter_resultat_camera(self, intent_data):
        """Gère l'extraction du bitmap, son redimensionnement et sa sauvegarde disque."""
        print("[CAMERA] Extraction du bitmap...")

        bitmap = None
        scaled_bitmap = None
        temp_path = None
        output = None

        try:
            # Extraction du petit aperçu (thumbnail) fourni par défaut par l'Intent Camera
            extras = intent_data.getExtras()
            bitmap = extras.get("data") if extras else None

            if bitmap is None:
                # Erreur si l'image n'est pas trouvée dans les extras
                couleur = "ff0000"
                texte = "PAS DE BITMAP..."
                self.texte_base = texte
                self.couleur = couleur
                self.afficher_message_temporaire(texte, couleur)
                return

            # --- 1. Redimensionnement pour économiser la RAM ---
            width = bitmap.getWidth()
            height = bitmap.getHeight()
            max_dimension = 2000

            # Vérification si l'image dépasse la limite de sécurité
            if width > max_dimension or height > max_dimension:
                scale = max_dimension / float(max(width, height))
                new_width = int(width * scale)
                new_height = int(height * scale)
                # Création d'une version réduite
                scaled_bitmap = Bitmap.createScaledBitmap(
                    bitmap, new_width, new_height, True
                )
                # Libération immédiate de la mémoire de l'original
                bitmap.recycle()
                bitmap = None
            else:
                scaled_bitmap = bitmap
                bitmap = None

            # --- 2. Sauvegarde sur le disque ---
            temp_dir = tempfile.gettempdir()
            # Génération d'un nom de fichier unique basé sur le timestamp
            temp_path = os.path.join(temp_dir, f"fusee_{int(time.time() * 1000)}.jpg")

            # Ouverture d'un flux d'écriture vers le fichier
            output = FileOutputStream(temp_path)
            # Compression agressive (40%) pour optimiser le traitement ultérieur
            result = scaled_bitmap.compress(CompressFormat.JPEG, 40, output)
            output.flush()
            output.close()
            output = None

            if not result:
                # Erreur si la compression a échoué
                couleur = "ff0000"
                texte = "ERREUR COMPRESSION..."
                self.texte_base = texte
                self.couleur = couleur
                self.afficher_message_temporaire(texte, couleur)
                return

            # --- 3. Vérification de l'existence du fichier ---
            time.sleep(0.5)

            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                couleur = "ff0000"
                texte = "PHOTO INTROUVABLE..."
                self.texte_base = texte
                self.couleur = couleur
                self.afficher_message_temporaire(texte, couleur)
                return

            print(f"[CAMERA] Fichier sauvegardé: {temp_path}")

            # --- 4. Nettoyage mémoire ---
            scaled_bitmap.recycle()
            scaled_bitmap = None
            gc.collect()

            # --- 5. Validation par Pillow ---
            self._traiter_image_valider_chemin(temp_path)

        except Exception as e:
            # Gestion des erreurs système graves lors de la manipulation du bitmap
            print(f"[CAMERA] ERREUR CRITIQUE: {e}")
            import traceback

            print(traceback.format_exc())
            couleur = "ff0000"
            texte = "ERREUR CRITIQUE..."
            self.texte_base = texte
            self.couleur = couleur
            self.afficher_message_temporaire(texte, couleur)

        finally:
            # Garantie de fermeture des flux et libération des ressources Java
            if output:
                try:
                    output.close()
                except:
                    pass
            if bitmap:
                try:
                    bitmap.recycle()
                except:
                    pass
            if scaled_bitmap:
                try:
                    scaled_bitmap.recycle()
                except:
                    pass
            gc.collect()

    def _traiter_image_valider_chemin(self, file_path):
        """Valide l'intégrité de l'image sélectionnée ou capturée."""
        print(f"[IMAGE] Validation du fichier: {file_path}")

        try:
            # Vérification de l'existence physique
            if not file_path or not os.path.exists(file_path):
                couleur = "ff0000"
                texte = "FORMAT INVALIDE..."
                self.texte_base = texte
                self.couleur = couleur
                self.afficher_message_temporaire(texte, couleur)
                return

            # Vérification que le fichier n'est pas corrompu (taille 0)
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                couleur = "ff0000"
                texte = "FICHIER VIDE..."
                self.texte_base = texte
                self.couleur = couleur
                self.afficher_message_temporaire(texte, couleur)
                try:
                    os.remove(file_path)
                except:
                    pass
                return

            # Tentative d'ouverture réelle avec Pillow pour tester le décodage
            with Image.open(file_path) as img:
                img.load()
                if img.format is None:
                    couleur = "ff0000"
                    texte = "FORMAT INVALIDE..."
                    self.texte_base = texte
                    self.couleur = couleur
                    self.afficher_message_temporaire(texte, couleur)
                    return

            # Si tout est OK, stockage du chemin et mise à jour de l'UI
            self.chemin_image_fusee = file_path
            couleur = "00ff00"
            texte = "FUSÉE SCANNÉE !"
            self.texte_base = texte
            self.couleur = couleur
            self.afficher_message_temporaire(texte, couleur)
            self.fusee_scannee = 1
            print("[IMAGE] Image validée et chemin stocké.")

        except Exception as e:
            # Erreur si le fichier n'est pas une image valide
            print(f"[IMAGE] ERREUR de validation: {e}")
            couleur = "ff0000"
            texte = "ERREUR TRAITEMENT IMAGE..."
            self.texte_base = texte
            self.couleur = couleur
            self.afficher_message_temporaire(texte, couleur)
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
        finally:
            gc.collect()

    def lancer_calcul_stabilite(self, instance):
        """Initialise et lance le processus d'extraction des dimensions."""
        # Vérification qu'une image est bien présente
        if self.chemin_image_fusee is None:
            couleur = "ff0000"
            texte = "AUCUNE FUSÉE SCANNÉE..."
            self.texte_base = texte
            self.couleur = couleur
            self.afficher_message_temporaire(texte, couleur)
            return

        # Vérification que l'image n'a pas été supprimée entre temps
        if not os.path.exists(self.chemin_image_fusee):
            couleur = "ff0000"
            texte = "IMAGE INTROUVABLE..."
            self.texte_base = texte
            self.couleur = couleur
            self.afficher_message_temporaire(texte, couleur)
            self.chemin_image_fusee = None
            return

        # Affichage du statut d'attente
        self.infos_label.text = "[i][color=000000]En cours...[/color][/i]"

        # Initialisation du flag de synchronisation sous forme de liste pour mutabilité
        self.calcul_termine_flag = [False]

        # Affichage de la fenêtre de progression
        self._afficher_popup_progression()

        # Création et démarrage du thread pour ne pas bloquer l'interface
        calcul_thread = threading.Thread(target=self._calcul_stabilite_thread)
        calcul_thread.daemon = True  # Le thread s'arrête si l'app est fermée
        calcul_thread.start()

    def _afficher_popup_progression(self):
        """Crée et affiche une fenêtre modale avec une barre de progression."""
        # Création du conteneur principal du popup
        root = BoxLayout(
            orientation="vertical", padding=self.pad(30), spacing=self.pad(20)
        )

        # ---------- Contenu ----------
        # Grille pour organiser les éléments de progression
        content = GridLayout(cols=1, spacing=self.pad(20), size_hint_y=None)
        # Ajustement dynamique de la hauteur de la grille
        content.bind(minimum_height=content.setter("height"))

        # Label indiquant l'action en cours
        label_calcul = Label(
            text="[b]Extraction en cours...[/b]",
            font_size=self.sp(27),
            markup=True,
            size_hint_y=None,
            height=self.sp(50),
            color=(0, 0, 0, 1),
        )
        content.add_widget(label_calcul)

        # Barre de progression visuelle
        progress_bar = ProgressBar(
            max=100, value=0, size_hint_y=None, height=self.sp(40)
        )
        content.add_widget(progress_bar)

        # Label affichant le pourcentage textuel
        label_pourcentage = Label(
            text="0 %",
            font_size=self.sp(25),
            size_hint_y=None,
            height=self.sp(40),
            color=(0, 0, 0, 1),
        )
        content.add_widget(label_pourcentage)

        # Ajout du contenu au layout racine
        root.add_widget(content)

        # Instanciation du Popup Kivy
        popup = Popup(
            title="Traitement",
            title_size=self.sp(25),
            #separator_height=0,
            content=root,
            size_hint=(0.8, None),
            auto_dismiss=False,
        )

        # Calcul final de la mise en page
        content.do_layout()
        # Calcul de la hauteur totale du popup en fonction du contenu
        popup.height = content.height + self.sp(180)

        # Ouverture de la fenêtre
        popup.open()

        # Variable mutable pour suivre le temps dans la fonction de rappel
        temps_ecoule = [0]

        def update_progress(dt):
            """Met à jour l'affichage de la progression à intervalles réguliers."""
            if not self.calcul_termine_flag[0]:
                # Incrémentation du temps écoulé
                temps_ecoule[0] += dt
                # Calcul du pourcentage basé sur le temps estimé
                p = min(100, (temps_ecoule[0] / self.temps) * 100)
                # Mise à jour des valeurs des widgets
                progress_bar.value = p
                label_pourcentage.text = f"{int(p)} %"
            else:
                # Forçage à 100% une fois le thread terminé
                progress_bar.value = 100
                label_pourcentage.text = "100 %"
                # Fermeture différée pour laisser l'utilisateur voir la fin
                Clock.schedule_once(lambda dt: popup.dismiss(), 1)
                return False

        # Planification de la mise à jour toutes les 100ms
        Clock.schedule_interval(update_progress, 0.1)

        # Sauvegarde de la référence du popup
        self.popup_progression = popup

    def _calcul_stabilite_thread(self):
        """Exécute les algorithmes lourds d'extraction d'image en arrière-plan."""
        try:
            # Ouverture de l'image pour analyse
            img = Image.open(self.chemin_image_fusee)

            # Appel au module métier pour extraire les contours et dimensions
            self.donnees_completes, image_cone = shapeExtraction.dimensions(img)
            import numpy as np

            # Libération immédiate de l'image et appel du GC
            img = None
            gc.collect()

            # Analyse spécifique de la pointe (ogive)
            self.donnees_completes["Forme_ogive"] = noseConeShape.noseConeType(
                image_cone
            )
            # Récupération des valeurs saisies dans l'UI
            self.donnees_completes["Q_ail"] = int(self.aileron_bas_spinner.text)
            self.donnees_completes["Q_can"] = 0

            # Fonction interne pour notifier le succès sur le thread principal
            def update_ui_success(dt):
                couleur = "00ff00"
                texte = "DONNÉES EXTRAITES !"
                self.fusee_dimensions = 1
                self.texte_base = texte
                self.couleur = couleur
                self.infos_label.text = f"[b][color={couleur}]{texte}[/color][/b]"

            # Planification de la mise à jour UI
            Clock.schedule_once(update_ui_success, 0)

        except Exception as e:
            # Gestion des erreurs durant l'analyse d'image
            print(f"ERREUR CALCUL STABILITÉ: {e}")
            import traceback

            print(traceback.format_exc())

            # Notification de l'échec sur l'interface
            def update_ui_error(dt):
                couleur = "ff0000"
                texte = "ERREUR EXTRACTION..."
                self.texte_base = texte
                self.couleur = couleur
                self.infos_label.text = f"[b][color={couleur}]{texte}[/color][/b]"

            Clock.schedule_once(update_ui_error, 0)

        finally:
            # Indication au popup que le travail est fini, quoi qu'il arrive
            self.calcul_termine_flag[0] = True

    def afficher_details(self, instance):
        """Déclenche l'affichage du récapitulatif des mesures."""
        # Animation d'attente
        self.infos_label.text = "[i][color=000000]En cours...[/color][/i]"

        # Sécurité : vérifier que le calcul a bien eu lieu
        if self.fusee_dimensions != 1:
            couleur = "ff0000"
            texte = "DIMENSIONS NON EXTRAITES..."
            self.texte_base = texte
            self.couleur = couleur
            Clock.schedule_once(
                lambda dt: setattr(
                    self.infos_label, "text", f"[b][color={couleur}]{texte}[/color][/b]"
                ),
                0.6,
            )
            return

        # Affichage du popup après un court délai visuel
        Clock.schedule_once(lambda dt: self._afficher_popup_details(), 0.6)

    def _afficher_popup_details(self):
        """Construit dynamiquement le contenu du popup de résultats."""

        # Rafraîchissement des quantités d'ailerons
        self.donnees_completes["Q_ail"] = int(self.aileron_bas_spinner.text)
        self.donnees_completes["Q_can"] = 0
        # Création du conteneur en grille simple colonne
        content = GridLayout(cols=1, spacing=self.pad(10), padding=self.pad(10))
        self.afficher_det = 1

        # --- Section Ogive ---
        section_ogive = Label(
            text="[b][u]OGIVE[/u][/b]",
            font_size=self.sp(34),
            color=(0, 0, 0, 1),
            size_hint_y=None,
            height=self.sp(50),
            markup=True,
        )
        content.add_widget(section_ogive)

        # Traduction de l'indice de forme en texte clair
        forme_dict = {0: "Parabolique", 1: "Ogivale", 2: "Conique"}
        forme_text = forme_dict.get(
            self.donnees_completes.get("Forme_ogive", 0), "Inconnue"
        )

        # Liste des données à afficher pour l'ogive
        ogive_data = [
            ("Longueur ogive", self.donnees_completes.get("Long_ogive", "N/A"), "mm"),
            ("Diamètre ogive", self.donnees_completes.get("D_og", "N/A"), "mm"),
            ("Forme ogive", forme_text, ""),
        ]

        # Ajout itératif des labels de données
        for label_text, value, unit in ogive_data:
            data_label = Label(
                text=f"{label_text} : [b]{value} {unit}[/b]",
                font_size=self.sp(27),
                color=(0, 0, 0, 1),
                size_hint_y=None,
                height=self.sp(35),
                markup=True,
                halign="left",
                valign="middle",
            )
            data_label.bind(size=data_label.setter("text_size"))
            content.add_widget(data_label)

        # Espacement visuel
        separateur1 = Label(text="", size_hint_y=None, height=self.sp(10))
        content.add_widget(separateur1)

        # --- Section Corps et Transitions ---
        section_transitions = Label(
            text="[b][u]CORPS[/u][/b]",
            font_size=self.sp(34),
            color=(0, 0, 0, 1),
            size_hint_y=None,
            height=self.sp(50),
            markup=True,
        )
        content.add_widget(section_transitions)

        # Données physiques globales de la structure
        general_data = [
            ("Longueur totale", self.donnees_completes.get("Long_tot", "N/A"), "mm"),
        ]

        for label_text, value, unit in general_data:
            data_label = Label(
                text=f"{label_text} : [b]{value} {unit}[/b]",
                font_size=self.sp(27),
                color=(0, 0, 0, 1),
                size_hint_y=None,
                height=self.sp(35),
                markup=True,
                halign="left",
                valign="middle",
            )
            data_label.bind(size=data_label.setter("text_size"))
            content.add_widget(data_label)

        # Gestion dynamique de l'affichage des transitions (si présentes)
        nb_trans = self.donnees_completes.get("Nb_trans", 0)
        if nb_trans >= 1:
            trans1_label = Label(
                text="[b]Transition 1 :[/b]",
                font_size=self.sp(29),
                color=(0.2, 0.2, 0.8, 1),
                size_hint_y=None,
                height=self.sp(40),
                markup=True,
            )
            content.add_widget(trans1_label)

            trans1_data = [
                ("  Longueur", self.donnees_completes.get("l_j", "N/A"), "mm"),
                ("  Diamètre haut", self.donnees_completes.get("D1j", "N/A"), "mm"),
                ("  Diamètre bas", self.donnees_completes.get("D2j", "N/A"), "mm"),
                ("  Position", self.donnees_completes.get("X_j", "N/A"), "mm"),
            ]

            for label_text, value, unit in trans1_data:
                data_label = Label(
                    text=f"{label_text} : [b]{value} {unit}[/b]",
                    font_size=self.sp(27),
                    color=(0, 0, 0, 1),
                    size_hint_y=None,
                    height=self.sp(32),
                    markup=True,
                    halign="left",
                    valign="middle",
                )
                data_label.bind(size=data_label.setter("text_size"))
                content.add_widget(data_label)

        # Répétition de la logique pour les transitions 2 et 3
        if nb_trans >= 2:
            trans2_label = Label(
                text="[b]Transition 2 :[/b]",
                font_size=self.sp(29),
                color=(0.2, 0.2, 0.8, 1),
                size_hint_y=None,
                height=self.sp(40),
                markup=True,
            )
            content.add_widget(trans2_label)

            trans2_data = [
                ("  Longueur", self.donnees_completes.get("l_r", "N/A"), "mm"),
                ("  Diamètre haut", self.donnees_completes.get("D1r", "N/A"), "mm"),
                ("  Diamètre bas", self.donnees_completes.get("D2r", "N/A"), "mm"),
                ("  Position", self.donnees_completes.get("X_r", "N/A"), "mm"),
            ]

            for label_text, value, unit in trans2_data:
                data_label = Label(
                    text=f"{label_text} : [b]{value} {unit}[/b]",
                    font_size=self.sp(27),
                    color=(0, 0, 0, 1),
                    size_hint_y=None,
                    height=self.sp(32),
                    markup=True,
                    halign="left",
                    valign="middle",
                )
                data_label.bind(size=data_label.setter("text_size"))
                content.add_widget(data_label)

        if nb_trans >= 3:
            trans3_label = Label(
                text="[b]Transition 3 :[/b]",
                font_size=self.sp(29),
                color=(0.2, 0.2, 0.8, 1),
                size_hint_y=None,
                height=self.sp(40),
                markup=True,
            )
            content.add_widget(trans3_label)

            trans3_data = [
                ("  Longueur", self.donnees_completes.get("l_s", "N/A"), "mm"),
                ("  Diamètre haut", self.donnees_completes.get("D1s", "N/A"), "mm"),
                ("  Diamètre bas", self.donnees_completes.get("D2s", "N/A"), "mm"),
                ("  Position", self.donnees_completes.get("X_s", "N/A"), "mm"),
            ]

            for label_text, value, unit in trans3_data:
                data_label = Label(
                    text=f"{label_text} : [b]{value} {unit}[/b]",
                    font_size=self.sp(27),
                    color=(0, 0, 0, 1),
                    size_hint_y=None,
                    height=self.sp(32),
                    markup=True,
                    halign="left",
                    valign="middle",
                )
                data_label.bind(size=data_label.setter("text_size"))
                content.add_widget(data_label)

        separateur2 = Label(text="", size_hint_y=None, height=self.sp(10))
        content.add_widget(separateur2)

        # --- Section Ailerons ---
        section_ailerons = Label(
            text="[b][u]AILERONS[/u][/b]",
            font_size=self.sp(34),
            color=(0, 0, 0, 1),
            size_hint_y=None,
            height=self.sp(50),
            markup=True,
        )
        content.add_widget(section_ailerons)

        # Affichage des dimensions géométriques des ailerons bas
        q_ail = int(self.aileron_bas_spinner.text)

        if q_ail > 0:
            ail_bas_data = [
                ("  Nombre d'ailerons", q_ail, ""),
                ("  Emplanture (m)", self.donnees_completes.get("m_ail", "N/A"), "mm"),
                ("  Saumon (n)", self.donnees_completes.get("n_ail", "N/A"), "mm"),
                ("  Flèche (p)", self.donnees_completes.get("p_ail", "N/A"), "mm"),
                ("  Envergure (E)", self.donnees_completes.get("E_ail", "N/A"), "mm"),
            ]

            for label_text, value, unit in ail_bas_data:
                data_label = Label(
                    text=f"{label_text} : [b]{value} {unit}[/b]",
                    font_size=self.sp(27),
                    color=(0, 0, 0, 1),
                    size_hint_y=None,
                    height=self.sp(32),
                    markup=True,
                    halign="left",
                    valign="middle",
                )
                data_label.bind(size=data_label.setter("text_size"))
                content.add_widget(data_label)

        # (Note: La partie ailerons haut/canard est présente dans le code mais non ajoutée au widget)
        q_can = int(self.aileron_haut_spinner.text)

        separateur3 = Label(text="", size_hint_y=None, height=self.sp(10))
        content.add_widget(separateur3)

        # Ajout d'un espace vide pour pousser le bouton vers le bas
        spacer = Label(text="", size_hint_y=1)
        content.add_widget(spacer)

        # Bouton pour refermer le récapitulatif
        btn_fermer = RoundedButton(
            text="Fermer",
            size_hint_y=None,
            height=self.sp(60),
            background_color=(0.8, 0.8, 0.8, 1),
            font_size=self.sp(25),
            color=(0, 0, 0, 1),
        )
        content.add_widget(btn_fermer)

        # Création de la fenêtre popup avec calcul de hauteur automatique
        popup = Popup(
            title="Résultats détaillés",
            content=content,
            size_hint=(0.9, None),
            title_size=self.sp(25),
        )
        content.do_layout()
        # Ajustement de la hauteur du popup selon le contenu sans dépasser l'écran
        popup.height = min(content.minimum_height + self.sp(130), Window.height * 0.95)

        # Mise à jour du statut global pour l'envoi
        couleur = "00ff00"
        texte = "PRÊT POUR L'ENVOI !"
        self.texte_base = texte
        self.couleur = couleur
        self.infos_label.text = f"[b][color={couleur}]{texte}[/color][/b]"

        btn_fermer.bind(on_release=popup.dismiss)
        popup.open()

    def send_api(self, instance):
        """Valide les conditions et transmet les données à l'API SPOCK."""
        # Vérification si le nombre d'ailerons a été modifié après extraction
        if self.donnees_completes["Q_ail"] != int(self.aileron_bas_spinner.text):
            self.afficher_det = 0

        # Arbre de décision pour valider l'état de l'application avant envoi
        if self.fusee_scannee:
            if self.fusee_dimensions:
                if self.afficher_det:
                    # Vérification de la présence des identifiants obligatoires
                    if self.SPOCK_id != "" and self.SPOCK_key != "":
                        # Appel effectif du module API
                        sended_to_SPOCK = spockApi.send_params_to_SPOCK(
                            self.SPOCK_id, self.SPOCK_key, self.donnees_completes
                        )
                        if sended_to_SPOCK:
                            couleur = "00ff00"
                            texte = "DONNÉES ENVOYÉES !"
                        else:
                            couleur = "ff0000"
                            texte = "PROBLÈME LORS DE L'ENVOI..."
                    else:
                        # Gestion précise des identifiants manquants
                        couleur = "ff0000"
                        if self.SPOCK_id == "" and self.SPOCK_key != "":
                            texte = "PAS D'ID RENSEIGNÉ..."
                        elif self.SPOCK_id != "" and self.SPOCK_key == "":
                            texte = "PAS DE KEY RENSEIGNÉE..."
                        elif self.SPOCK_id == "" and self.SPOCK_key == "":
                            texte = "PAS D'ID ET DE KEY RENSEIGNÉS..."
                else:
                    # L'utilisateur doit vérifier les résultats avant d'autoriser l'envoi
                    couleur = "ff0000"
                    texte = "VÉRIFIEZ LES RÉSULTATS..."
            else:
                couleur = "ff0000"
                texte = "DIMENSIONS NON EXTRAITES..."
        else:
            couleur = "ff0000"
            texte = "AUCUNE FUSÉE SCANNÉE..."
            self.texte_base = texte
            self.couleur = couleur
        # Affichage du statut final après tentative d'envoi
        self.afficher_message_temporaire(texte, couleur, 1.2)

    def afficher_guide(self, instance):
        """Lance l'affichage du guide d'utilisation avec animation."""
        self.infos_label.text = "[i][color=000000]En cours...[/color][/i]"
        Clock.schedule_once(lambda dt: self._afficher_popup_guide(), 0.6)

    def _afficher_popup_guide(self):
        """Construit le contenu textuel et illustré du guide d'utilisation."""

        # Conteneur scrollable car le guide est long
        content = GridLayout(
            cols=1, spacing=self.pad(10), padding=self.pad(10), size_hint_y=None
        )
        content.bind(minimum_height=content.setter("height"))

        # --- Section : Utilisation ---
        section_utilisation = Label(
            text="[b][u]UTILISATION DE VULCAIN[/u][/b]",
            font_size=self.sp(34),
            color=(0, 0, 0, 1),
            size_hint_y=None,
            height=self.sp(50),
            markup=True,
        )
        content.add_widget(section_utilisation)

        # Texte descriptif des étapes (BBcode Kivy utilisé)
        texte_utilisation = Label(
            text="1- Sélectionner le nombre d'ailerons.\n\n"
            '2- Cliquez sur [b]"Scanner fusée"[/b]. Choisissez comment vous voulez importer l\'image de votre fusée (prendre une photo ou choisir depuis la galerie).\n\n'
            "3- Cliquez sur [b]\"Extraire les dimensions\"[/b] pour lancer l'extraction des dimensions de la fusée à partir de l'image.\n\n"
            '4- Cliquez sur [b]"Afficher les résultats"[/b] pour vérifier leur cohérence.\n\n'
            '5- Cliquez sur [b]"Identification SPOCK"[/b] pour s\'assurer que l\'ID et la KEY ont bien été importés depuis SPOCK. Si ce n\'est pas le cas, vous pouvez les entrer manuellement sans oublier de cliquer sur [b]"Sauvegarder les identifiants"[/b] une fois cela fait.\n\n'
            '6- Cliquez sur [b]"Envoyer les données"[/b] pour transférer les résultats sur SPOCK.\n',
            font_size=self.sp(27),
            color=(0, 0, 0, 1),
            size_hint_y=None,
            halign="left",
            valign="top",
            markup=True,
            text_size=(Window.width * 0.75, None),
        )
        texte_utilisation.bind(texture_size=texte_utilisation.setter("size"))
        content.add_widget(texte_utilisation)

        # Message d'erreur/conseil
        texte_erreur = Label(
            text='[i][color=cc0000]En cas d\'erreur ou si vous souhaitez recommencer, cliquez sur [b]"Réinitialiser"[/b] et reprennez les étapes depuis le point 1.[/i][/color]',
            font_size=self.sp(27),
            color=(0, 0, 0, 1),
            size_hint_y=None,
            halign="center",
            valign="top",
            markup=True,
            text_size=(Window.width * 0.75, None),
        )
        texte_erreur.bind(texture_size=texte_erreur.setter("size"))
        content.add_widget(texte_erreur)

        separateur1 = Label(text="", size_hint_y=None, height=self.sp(20))
        content.add_widget(separateur1)

        # --- Section : Prise de vue ---
        section_prise_vue = Label(
            text="[b][u]PRISE DE VUE[/u][/b]",
            font_size=self.sp(34),
            color=(0, 0, 0, 1),
            size_hint_y=None,
            height=self.sp(50),
            markup=True,
        )
        content.add_widget(section_prise_vue)

        texte_prise_vue = Label(
            text="Pour obtenir le meilleur résultat possible, merci de faire en sorte que votre photo respecte les points suivants :\n\n"
            "• Fond uni et lisse d'une couleur différente de celles présentes sur la fusée\n\n"
            "• Les 4 mires doivent être complètement visibles\n\n"
            "• Cadrez l'image de telle sorte à ce que le fond soit le même partout\n\n"
            "• Fusée centrée et bien verticale\n\n"
            "• Faire en sorte qu'il y ait au moins un aileron dont la surface est parallèle avec le fond",
            font_size=self.sp(27),
            color=(0, 0, 0, 1),
            size_hint_y=None,
            halign="left",
            valign="top",
            markup=True,
            text_size=(Window.width * 0.75, None),
        )
        texte_prise_vue.bind(texture_size=texte_prise_vue.setter("size"))
        content.add_widget(texte_prise_vue)

        espace_images = Label(text="", size_hint_y=None, height=self.sp(15))
        content.add_widget(espace_images)

        # Disposition horizontale des images d'exemple
        images_layout = BoxLayout(
            orientation="horizontal",
            spacing=self.pad(10),
            size_hint_y=None,
            height=self.sp(400),
        )

        # Conteneur pour l'exemple "Bonne prise de vue"
        image1_container = BoxLayout(orientation="vertical", spacing=self.pad(5))
        image1_label = Label(
            text="[b][color=00AA00]Bonne[/color][/b]",
            font_size=self.sp(27),
            size_hint_y=None,
            height=self.sp(40),
            markup=True,
        )
        try:
            image1 = KivyImage(
                source="guide_1.jpg", size_hint=(1, 1), fit_mode="contain"
            )
        except Exception:
            image1 = Label(
                text="[Image 1\nnon trouvée]",
                font_size=self.sp(27),
                color=(0.5, 0.5, 0.5, 1),
            )

        image1_container.add_widget(image1_label)
        image1_container.add_widget(image1)
        images_layout.add_widget(image1_container)

        # Conteneur pour l'exemple "Mauvaise prise de vue"
        image2_container = BoxLayout(orientation="vertical", spacing=self.pad(5))
        image2_label = Label(
            text="[b][color=FF0000]Mauvaise[/color][/b]",
            font_size=self.sp(27),
            size_hint_y=None,
            height=self.sp(40),
            markup=True,
        )
        try:
            image2 = KivyImage(
                source="guide_2.jpg", size_hint=(1, 1), fit_mode="contain"
            )
        except Exception:
            image2 = Label(
                text="[Image 2\nnon trouvée]",
                font_size=self.sp(27),
                color=(0.5, 0.5, 0.5, 1),
            )

        image2_container.add_widget(image2_label)
        image2_container.add_widget(image2)
        images_layout.add_widget(image2_container)

        content.add_widget(images_layout)

        separateur2 = Label(text="", size_hint_y=None, height=self.sp(20))
        content.add_widget(separateur2)

        # --- Section : Conseils Techniques ---
        section_conseils = Label(
            text="[b][u]CONSEILS[/u][/b]",
            font_size=self.sp(34),
            color=(0, 0, 0, 1),
            size_hint_y=None,
            height=self.sp(50),
            markup=True,
        )
        content.add_widget(section_conseils)

        texte_conseils = Label(
            text="• Privilégiez l'utilisation de VULCAIN [b]sans le mode économie d'énergie[/b] de votre téléphone activé pour de meilleures performances.\n\n"
            "• Si vous prenez une photo depuis VULCAIN, inutile d'utiliser la résolution maximale de votre appareil.\n",
            font_size=self.sp(27),
            color=(0, 0, 0, 1),
            size_hint_y=None,
            halign="left",
            valign="top",
            markup=True,
            text_size=(Window.width * 0.75, None),
        )
        texte_conseils.bind(texture_size=texte_conseils.setter("size"))
        content.add_widget(texte_conseils)

        texte_conseils2 = Label(
            text="[b][i][color=cc0000]Le mode économie d'énergie activé + une résolution d'image élevée peut provoquer des crashes si vous prenez une photo depuis l'application. Si cela se produit, veuillez appliquer les deux points précédents.[/i][/b][/color]",
            font_size=self.sp(27),
            color=(0, 0, 0, 1),
            size_hint_y=None,
            halign="center",
            valign="top",
            markup=True,
            text_size=(Window.width * 0.75, None),
        )
        texte_conseils2.bind(texture_size=texte_conseils2.setter("size"))
        content.add_widget(texte_conseils2)

        espace_final = Label(text="", size_hint_y=None, height=self.sp(15))
        content.add_widget(espace_final)

        # Bouton de fermeture
        btn_fermer = RoundedButton(
            text="Fermer",
            size_hint_y=None,
            height=self.sp(60),
            background_color=(0.8, 0.8, 0.8, 1),
            font_size=self.sp(25),
            color=(0, 0, 0, 1),
        )
        content.add_widget(btn_fermer)

        # Inclusion de la grille dans un ScrollView
        scroll_view = ScrollView()
        scroll_view.add_widget(content)

        # Création du popup guide
        popup = Popup(
            title="Guide VULCAIN",
            content=scroll_view,
            size_hint=(0.9, 0.9),
            title_size=self.sp(25),
        )

        # Restauration du label d'info
        self.infos_label.text = (
            f"[b][color={self.couleur}]{self.texte_base}[/color][/b]"
        )

        btn_fermer.bind(on_release=popup.dismiss)
        popup.open()

    def afficher_identifiants_SPOCK(self, instance):
        """Crée une interface de saisie pour configurer manuellement SPOCK."""

        # Grille de saisie
        content = GridLayout(cols=1, spacing=self.pad(10), padding=self.pad(10))

        # --- Champ ID ---
        id_label = Label(
            text="ID :", font_size=self.sp(27), color=(0, 0, 0, 1), size_hint_y=0.1
        )
        content.add_widget(id_label)
        id_input = TextInput(
            text=self.SPOCK_id,
            font_size=self.sp(27),
            multiline=False,
            size_hint_y=0.1,
            foreground_color=(0, 0, 0, 1),
        )
        content.add_widget(id_input)

        # --- Champ Key ---
        key_label = Label(
            text="Key :", font_size=self.sp(27), color=(0, 0, 0, 1), size_hint_y=0.1
        )
        content.add_widget(key_label)
        key_input = TextInput(
            text=self.SPOCK_key,
            font_size=self.sp(27),
            multiline=False,
            size_hint_y=0.1,
            foreground_color=(0, 0, 0, 1),
        )
        content.add_widget(key_input)

        # --- Champ Code ---
        code_label = Label(
            text="Code (optionnel) :",
            font_size=self.sp(27),
            color=(0, 0, 0, 1),
            size_hint_y=0.1,
        )
        content.add_widget(code_label)
        code_input = TextInput(
            text=self.SPOCK_code,
            font_size=self.sp(27),
            multiline=False,
            size_hint_y=0.1,
            foreground_color=(0, 0, 0, 1),
        )
        content.add_widget(code_input)

        # Bouton de sauvegarde locale
        btn_sauvegarder = RoundedButton(
            text="Sauvegarder les identifiants",
            size_hint_y=0.1,
            background_color=(0.2, 0.6, 1, 1),
            font_size=self.sp(25),
            color=(1, 1, 1, 1),
        )
        content.add_widget(btn_sauvegarder)

        # Zone de feedback
        message_label = Label(
            text="",
            font_size=self.sp(27),
            color=(0, 0.6, 0, 1),
            size_hint_y=0.1,
            markup=True,
        )
        content.add_widget(message_label)

        # Bouton pour redirection web
        btn_ouvrir_SPOCK = RoundedButton(
            text="Ouvrir la page SPOCK",
            size_hint_y=0.1,
            background_color=(0.8, 0.4, 0, 1),
            font_size=self.sp(25),
            color=(1, 1, 1, 1),
        )
        content.add_widget(btn_ouvrir_SPOCK)

        # Bouton fermeture
        btn_fermer = RoundedButton(
            text="Fermer",
            size_hint_y=0.1,
            background_color=(0.8, 0.8, 0.8, 1),
            font_size=self.sp(25),
            color=(0, 0, 0, 1),
        )
        content.add_widget(btn_fermer)

        popup = Popup(
            title="Identifiants SPOCK",
            content=content,
            size_hint=(0.9, 0.9),
            title_size=self.sp(25),
        )

        def sauvegarder_identifiants(btn_instance):
            """Action de sauvegarde des champs dans les variables de l'application."""
            message_label.text = "[i][color=000000]En cours...[/color][/i]"

            def executer_sauvegarde(dt):
                # Mise à jour visuelle du bouton et des données
                btn_instance._bg_color = (0.2, 0.6, 1, 1)
                btn_instance.bg_color.rgba = (0.2, 0.6, 1, 1)

                self.SPOCK_id = id_input.text.strip()
                self.SPOCK_key = key_input.text.strip()
                self.SPOCK_code = code_input.text.strip()
                # Synchronisation avec la mémoire "Intent"
                self.derniers_identifiants_recus["id"] = self.SPOCK_id
                self.derniers_identifiants_recus["key"] = self.SPOCK_key
                self.derniers_identifiants_recus["code"] = self.SPOCK_code
                message_label.text = (
                    "[b][color=00FF00]IDENTIFIANTS SAUVEGARDÉS ![/color][/b]"
                )

            Clock.schedule_once(executer_sauvegarde, 0.6)

        def ouvrir_SPOCK(btn_instance):
            """Action d'ouverture de l'URL SPOCK avec le code fourni."""
            message_label.text = "[i][color=000000]En cours...[/color][/i]"

            def executer_ouverture(dt):
                # Détermination du code à utiliser (priorité au champ texte)
                code_a_utiliser = (
                    self.SPOCK_code
                    if self.SPOCK_code != ""
                    else code_input.text.strip()
                )

                if code_a_utiliser == "":
                    # Erreur si aucun code n'est renseigné
                    btn_instance._bg_color = (0.8, 0.4, 0, 1)
                    btn_instance.bg_color.rgba = (0.8, 0.4, 0, 1)
                    message_label.text = "[b][color=FF0000]CODE MANQUANT...[/color][/b]"
                    return

                btn_instance._bg_color = (0.8, 0.4, 0, 1)
                btn_instance.bg_color.rgba = (0.8, 0.4, 0, 1)
                message_label.text = "[b][color=0000ff]OUVERTURE DE SPOCK ![/color][/b]"

                def ouvrir_url(dt):
                    # Construction de l'URL finale et lancement du navigateur
                    url = f"https://www.planete-sciences.org/espace/spock/stabtraj.html?code={code_a_utiliser}"
                    open_website(url)

                Clock.schedule_once(ouvrir_url, 1.0)

            Clock.schedule_once(executer_ouverture, 0.6)

        def ouvrir_SPOCK_et_sauvegarder(btn_instance):
            """Chaîne les deux actions de sauvegarde et d'ouverture."""
            sauvegarder_identifiants(btn_instance)
            ouvrir_SPOCK(btn_instance)

        # Liaison des fonctions aux boutons du popup
        btn_sauvegarder.bind(on_release=sauvegarder_identifiants)
        btn_ouvrir_SPOCK.bind(on_release=ouvrir_SPOCK_et_sauvegarder)
        btn_fermer.bind(on_release=popup.dismiss)

        popup.open()


# Lancement du programme
if __name__ == "__main__":
    # Démarrage de l'instance de l'application Vulcain
    VulcainApp().run()