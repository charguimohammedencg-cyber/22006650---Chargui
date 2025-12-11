📘 GRAND GUIDE : ANATOMIE D'UN PROJET DATA SCIENCE (Version Finance)
Ce document décortique le cycle de vie d'un projet de Machine Learning appliqué à la Banque. Il reprend la structure exacte de l'exemple médical pour faciliter la comparaison.

1. Le Contexte Métier et la Mission
Le Problème (Business Case)
Une banque perd de l'argent chaque fois qu'elle accorde un prêt à un client qui ne rembourse pas.

Objectif : Créer un "Algorithme de Scoring" pour décider automatiquement d'accorder ou refuser un crédit.

L'Enjeu critique : La matrice des coûts est asymétrique.

Refuser un bon client (Faux Positif) : La banque perd les intérêts du prêt (Manque à gagner).

Accepter un mauvais payeur (Faux Négatif) : La banque perd tout le capital prêté. C'est le risque majeur. L'IA doit être vigilante sur le risque.

Les Données (L'Input)
Nous simulons un dataset bancaire classique.

X (Features) : Caractéristiques financières (Revenu annuel, Ratio Dette/Revenu, Âge, Nombre de crédits en cours, etc.).

y (Target) : Binaire. 0 = Rembourse (Bon payeur), 1 = Défaut de paiement (Mauvais payeur).

2. Le Code Python (Laboratoire)
Ce script génère des données bancaires simulées et lance le pipeline complet.

Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')

# --- PHASE 1 : ACQUISITION & SIMULATION ---
# On génère un dataset synthétique pour l'exercice (1000 clients, 20 variables)
X_raw, y_raw = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                                   n_redundant=5, random_state=42, flip_y=0.05)

# On nomme quelques colonnes pour rendre ça concret
feature_names = [f"Feature_{i}" for i in range(20)]
feature_names[0] = "Revenu_Annuel"
feature_names[1] = "Dette_Totale"
feature_names[2] = "Age_Client"

df = pd.DataFrame(X_raw, columns=feature_names)
df['target'] = y_raw # 1 = Défaut, 0 = Rembourse

# Simulation de la réalité (Données sales)
np.random.seed(42)
df_dirty = df.copy()
# On imagine que 5% des clients n'ont pas déclaré leur revenu (NaN)
for col in df.columns[:-1]:
    df_dirty.loc[df_dirty.sample(frac=0.05).index, col] = np.nan

# --- PHASE 2 : DATA WRANGLING (NETTOYAGE) ---
X = df_dirty.drop('target', axis=1)
y = df_dirty['target']

# Stratégie d'imputation
imputer = SimpleImputer(strategy='median') 
# Note : En finance, on préfère souvent la médiane à la moyenne car les revenus sont très disparates.
X_imputed = imputer.fit_transform(X)
X_clean = pd.DataFrame(X_imputed, columns=X.columns)

# --- PHASE 3 : ANALYSE EXPLORATOIRE (EDA) ---
print("--- Statistiques Descriptives ---")
print(X_clean[['Revenu_Annuel', 'Dette_Totale', 'Age_Client']].describe())

# --- PHASE 4 : PROTOCOLE EXPÉRIMENTAL (SPLIT) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42
)

# --- PHASE 5 : INTELLIGENCE ARTIFICIELLE (RANDOM FOREST) ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- PHASE 6 : AUDIT DE PERFORMANCE ---
y_pred = model.predict(X_test)

print(f"\n--- Accuracy Globale : {accuracy_score(y_test, y_pred)*100:.2f}% ---")
print("\n--- Rapport Détaillé ---")
print(classification_report(y_test, y_pred, target_names=['Bon Payeur', 'Défaut']))

# Visualisation des erreurs
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Reds')
plt.title('Matrice de Confusion : Réalité vs IA (Risque)')
plt.ylabel('Vraie Situation')
plt.xlabel('Prédiction Banque')
plt.show()
3. Analyse Approfondie : Nettoyage (Data Wrangling)
Le Problème Mathématique du "Vide"
Comme pour le cancer, un dossier de prêt incomplet (ex: revenu manquant) fait planter le calcul.

La Mécanique de l'Imputation (Nuance Finance)
Ici, nous avons utilisé SimpleImputer(strategy='median') au lieu de la moyenne.

Pourquoi ? Les salaires suivent souvent une loi de Pareto (beaucoup de petits salaires, quelques milliardaires).

Si Jeff Bezos est dans votre base de données, la moyenne explose et ne représente plus le "client type". La médiane est insensible aux valeurs extrêmes (robuste aux outliers).

💡 Le Coin de l'Expert (Data Leakage)
Le même principe s'applique : calculer la médiane des revenus sur l'ensemble de la base avant le split est une "tricherie". Vous utilisez l'information des clients futurs (Test) pour estimer la richesse des clients actuels (Train).

4. Analyse Approfondie : Exploration (EDA)
Décrypter .describe()
Min/Max : En banque, cela permet de détecter des erreurs de saisie (ex: un âge négatif ou un revenu annuel de 0€ pour un prêt immobilier).

Ecart-type (Std) : Si l'écart-type de la colonne "Dette" est énorme, cela signifie que votre clientèle est très hétérogène.

La Multicollinéarité
Dans la finance, c'est fréquent.

Exemple : Revenu Annuel et Impôts Payés.

Ces deux variables racontent la même histoire. Si vous utilisez une Régression Logistique (fréquente en banque pour des raisons réglementaires d'explicabilité), vous devrez en supprimer une des deux pour éviter que le modèle ne "panique" mathématiquement. Le Random Forest, lui, gère ça très bien.

5. Analyse Approfondie : Méthodologie (Split)
Le Concept : Backtesting
En finance, le Test Set simule le "Backtesting". On se demande : "Si j'avais utilisé cet algorithme l'année dernière (sur des dossiers que je connais déjà), combien d'argent aurais-je perdu ?".

Les Paramètres
train_test_split(test_size=0.2)

On entraîne l'IA sur 800 dossiers historiques.

On la teste sur 200 dossiers "fermés" pour vérifier si elle aurait bien prédit les défauts de paiement qui ont réellement eu lieu.

6. FOCUS THÉORIQUE : L'Algorithme Random Forest 🌲
Pourquoi les banques aiment-elles le Random Forest (ou son cousin le XGBoost) ?

A. La Gestion des Non-Linéarités
Le risque de crédit n'est pas linéaire.

Avoir 20 ans et peu de revenus = Risqué.

Avoir 20 ans et beaucoup de revenus = Très bon client (Avenir prometteur).

Un modèle linéaire simple a du mal avec ces interactions "Si... Alors...". L'arbre de décision excelle ici.

B. La Force du Groupe (Bagging)
Bootstrapping : Chaque arbre s'entraîne sur un sous-groupe de clients. Certains arbres deviennent experts sur les "Jeunes Actifs", d'autres sur les "Retraités".

Feature Randomness : Certains arbres n'ont pas le droit de regarder le "Revenu". Ils doivent juger le client uniquement sur son "Historique de découvert". Cela crée des arbres très perspicaces sur les comportements bancaires, et pas juste sur la richesse.

C. Le Consensus
Si 70 arbres disent "Risque de défaut" et 30 disent "Client sûr", la banque refuse le prêt. C'est la sagesse de la foule.

7. Analyse Approfondie : Évaluation (ROI)
En banque, on ne parle pas juste de précision, mais de Coût du Risque.

A. La Matrice de Confusion (Quadrants Financiers)
Vrais Négatifs (TN) : Prédit Bon Payeur | Réel Bon Payeur. (La banque gagne des intérêts).

Vrais Positifs (TP) : Prédit Défaut | Réel Défaut. (La banque évite une perte, bravo).

Faux Positifs (FP) : Prédit Défaut | Réel Bon Payeur. (Occasion manquée, le client va à la concurrence).

Faux Négatifs (FN) : Prédit Bon Payeur | Réel Défaut.

Impact : Perte sèche du capital. C'est l'erreur la plus coûteuse.

B. Les Métriques Stratégiques
Précision (Precision) : "Fiabilité du refus".

TP/(TP+FP)
Parmi les gens que j'ai classés "Mauvais Payeurs", combien l'étaient vraiment ?

Le Rappel (Recall) : "Couverture du Risque".

TP/(TP+FN)
De tous les défauts de paiement qui ont eu lieu, combien en ai-je anticipé ?

Si le Recall est de 50%, votre banque est une passoire : elle ne voit pas venir la moitié des faillites personnelles.

Conclusion du Projet
Pour un projet bancaire, si le Random Forest donne un bon Recall mais rejette trop de bons clients (faible Précision), on peut ajuster le seuil de probabilité (ex: refuser le prêt si la probabilité de défaut > 30% au lieu de 50%) pour durcir la politique de risque. C'est là que la Data Science rencontre la Stratégie d'Entreprise.
