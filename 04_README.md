# 📰 Détection Automatique de Fake News — NLP & Machine Learning

> Projet de classification de texte — Comparaison de 4 modèles de Machine Learning pour détecter automatiquement les articles de désinformation à partir de leur contenu textuel.

---

## 🎯 Problématique

Avec l'explosion des réseaux sociaux et la rapidité de circulation de l'information, distinguer un vrai article journalistique d'un article fabriqué pour tromper est devenu un défi majeur. Ce projet automatise cette détection à partir du **contenu textuel des articles**, sans vérification humaine, en utilisant des techniques de NLP et de Machine Learning.

---

## 🗂️ Structure du projet

```
📁 fake_news_detection/
│
├── 📓 FakeNews_Detection_Sarra.ipynb   # Notebook principal — Pipeline complet
├── 🐍 fake_news_detection.py           # Script Python
│
├── 📊 Fake.csv                         # Articles de désinformation (label = 0)
├── 📊 True.csv                         # Articles réels (label = 1)
│
└── 📁 images/
    └── comparaison_modeles.png         # Comparaison des accuracies des modèles
```

📥 **Dataset :** [`Fake.csv`](https://github.com/sarra725/fake-news-detection/blob/main/01_data/Fake.csv) & [`True.csv`](https://github.com/sarra725/fake-news-detection/blob/main/01_data/True.csv) — Source : [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## ⚙️ Pipeline NLP

### 1️⃣ Chargement & Étiquetage
- `Fake.csv` → label `0` (faux)
- `True.csv` → label `1` (réel)
- Concaténation et mélange aléatoire (`random_state=42`)

### 2️⃣ Nettoyage du texte
Étape critique pour éviter le **data leakage** :
- Suppression des marqueurs sources (ex: `WASHINGTON (Reuters) -`)
- Conversion en minuscules
- Suppression des caractères non alphabétiques et des chiffres
- Normalisation des espaces

### 3️⃣ Vectorisation & Sélection de features
- **TF-IDF** avec `max_features=5000` et stop words anglais
- **SelectKBest (chi²)** pour garder les **1 000 mots** les plus discriminants
- Split **80/20** effectué **avant** la vectorisation pour éviter tout leakage

---

## 🤖 Modèles comparés

| Modèle | Description |
|---|---|
| **Logistic Regression** | Modèle linéaire de référence, rapide et interprétable |
| **Naive Bayes** | Classifieur probabiliste adapté au NLP |
| **Random Forest** | Ensemble d'arbres de décision (max_depth=20) |
| **LinearSVC** | SVM linéaire, très efficace sur matrices creuses TF-IDF |

📄 [`fake_news_detection.py`](https://github.com/sarra725/fake-news-detection/blob/main/02_scripts/FakeNews_Detection.ipynb)

---

## 📊 Résultats & Comparaison

| Modèle | Accuracy |
|:---|:---:|
| 🥉 Logistic Regression | 97.0% |
| Naive Bayes | 93.4% |
| 🥈 Random Forest | 97.3% |
| 🥇 **LinearSVC** | **97.6%** |

> 💡 **Conclusion :** Le **LinearSVC** obtient la meilleure accuracy (97.6%). Le **Naive Bayes** est le moins performant mais reste rapide. La **Logistic Regression** offre un excellent rapport performance/interprétabilité.

---

## 📈 Comparaison des Modèles

![Comparaison des accuracies des modèles](https://github.com/sarra725/fake-news-detection/blob/main/03_images/Comparaison%20des%20mod%C3%A9les.png)

> *Le graphique montre clairement que LinearSVC, Random Forest et Logistic Regression obtiennent des scores proches (97%+), tandis que Naive Bayes se distingue avec un score plus bas à 93.4%.*

---

## 🔍 Prédiction sur de nouveaux articles

Le pipeline expose une fonction `predict_news()` qui prend un texte brut et retourne le label **REAL** ou **FAKE** avec un score de confiance :

```python
label, confidence = predict_news("The Federal Reserve raised interest rates by 0.25 percent.")
# → ('REAL', 98.5)

label, confidence = predict_news("BREAKING: Hillary Clinton secretly arrested — Deep State exposed!")
# → ('FAKE', 99.1)
```

---

## 🚀 Installation & Utilisation

### Prérequis

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Lancer le script

```bash
python fake_news_detection.py
```

### Lancer le notebook

```bash
jupyter notebook FakeNews_Detection_Sarra.ipynb
```

> ⚠️ **Important :** Placer `Fake.csv` et `True.csv` à la racine du projet ou adapter le chemin :
> ```python
> fake = pd.read_csv("Fake.csv")
> true = pd.read_csv("True.csv")
> ```

---

## 📦 Dépendances

| Bibliothèque | Usage |
|---|---|
| `pandas` / `numpy` | Manipulation des données |
| `scikit-learn` | Vectorisation TF-IDF, modèles ML, évaluation |
| `matplotlib` / `seaborn` | Visualisation des résultats |
| `re` | Nettoyage des textes (expressions régulières) |

---

## 👩‍💻 Auteure

**Sarra** — Étudiante en NLP & Machine Learning  
Projet réalisé dans le cadre d'un cours de traitement automatique du langage naturel.
