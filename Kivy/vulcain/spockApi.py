import requests


def send_params_to_SPOCK(id_, key, donnees):
    """Transmet les paramètres finaux de la fusée au serveur SPOCK via une requête PUT."""
    # Construction de l'URL cible incluant les identifiants de sécurité (ID et KEY)
    url = f"https://www.planete-sciences.org/espace/spock/api.html?object=stabtraj&id={id_}&key={key}"

    # Construction de l'objet JSON (Data Frame) conforme aux attentes de l'API SPOCK
    DF = {
        "object": "stabtraj",  # Définit le type d'objet visé sur le serveur
        "id": id_,  # Identifiant de l'utilisateur/fusée
        "key": key,  # Clé d'authentification
        "data": donnees,  # Dictionnaire contenant toutes les mesures extraites
    }

    try:
        # Envoi de la requête PUT (Mise à jour de données) avec les données JSON
        r = requests.put(url, json=DF)
        # Affichage du code retour (ex: 200 pour succès)
        print("Envoi terminé :", r.status_code)

        # Retourne 1 (Succès) si le serveur a validé l'envoi
        if r.status_code == 200:
            return 1
        else:
            # Retourne 0 si le serveur a refusé les données
            return 0
    except Exception as e:
        # Journalisation de l'erreur réseau en cas d'échec de communication
        print("Erreur request :", e)
        return 0