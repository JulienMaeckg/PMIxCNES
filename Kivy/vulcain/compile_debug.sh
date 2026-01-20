#!/bin/bash

echo "Nettoyage complet..."
buildozer android clean

echo "Compilation de Vulcain en mode Debug..."

# Étape 1 : Buildozer prépare
echo "Préparation avec buildozer..."
buildozer android debug

# Étape 2 : Ajout du FileProvider
echo "Ajout du FileProvider au manifest..."
sed -i 's|</application>|    <provider android:name="androidx.core.content.FileProvider" android:authorities="${applicationId}.fileprovider" android:exported="false" android:grantUriPermissions="true"><meta-data android:name="android.support.FILE_PROVIDER_PATHS" android:resource="@xml/file_paths" /></provider>\n    </application>|' .buildozer/android/platform/build-arm64-v8a_armeabi-v7a/dists/vulcain/src/main/AndroidManifest.xml

# Étape 3 : Navigation vers le dossier de build
cd .buildozer/android/platform/build-arm64-v8a_armeabi-v7a/dists/vulcain

# Étape 4 : Compilation de l'APK debug
echo "Compilation de l'APK debug..."
./gradlew assembleDebug

# Étape 5 : Retour au dossier projet
cd ~/projets/vulcain

# Étape 6 : Création du dossier output
mkdir -p output

# Étape 7 : Copie de l'APK dans le dossier output
echo "Copie de l'APK dans le dossier output..."
cp .buildozer/android/platform/build-arm64-v8a_armeabi-v7a/dists/vulcain/build/outputs/apk/debug/vulcain-debug.apk output/

echo ""
echo "Compilation terminée !"
echo "APK debug disponible dans : ~/projets/vulcain/output/vulcain-debug.apk"
