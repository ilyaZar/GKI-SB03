# Entscheidungsbäume sind ein zentrales Werkzeug des maschinellen Lernens
# Sie sind leicht interpretierbar, visualisierbar und robust gegenüber fehlenden Werten
# Hier wird ein Entscheidungsbaum zur Klassifikation mit dem Iris-Datensatz aufgebaut

# 1. Importieren der benötigten Bibliothek zum Datenhandling
import pandas as pd

# 2. Einlesen des Datensatzes
# Die Datei muss im CSV-Format vorliegen, Semikolon als Spaltentrenner
# Pfad anpassen: z. B. 'daten/iris.csv' oder absoluter Pfad wie 'C:/Users/…'
file_path = "data/processed/iris.csv"
df = pd.read_csv(file_path, sep=",")

# 3. Aufteilen in Merkmale (X) und Zielvariable (y)
# Die Zielvariable 'species' soll vorhergesagt werden
# Alle anderen Spalten werden als Prädiktoren verwendet
features = [c for c in df.columns if c != "species"]
X = df[features]
y = df["species"]

# Der Iris-Datensatz stammt aus dem UCI Machine Learning Repository
# Er enthält Messdaten zu Kelch- und Blütenblättern dreier Iris-Arten
# Ziel ist die Vorhersage der Art ('species') anhand von vier Merkmalen:
# - sepal_length, sepal_width, petal_length, petal_width
#
# Die Daten werden aus einer CSV-Datei geladen:
# - 'file_path' muss auf die CSV-Datei zeigen
# - Trennzeichen ist Semikolon (;)
#
# Für die spätere Modellierung mit einem Entscheidungsbaum:
# - X enthält alle erklärenden Variablen
# - y enthält das Klassenlabel (species)


# Aufteilen der Daten in Trainings- und Testdaten
# Wichtig: Testdaten dürfen nicht im Training verwendet werden, um mögliches
# Overfitting zu erkennen
from sklearn.model_selection import train_test_split

# Die Funktion train_test_split mischt die Daten zufällig und teilt sie auf:
# - X_tr, y_tr: Trainingsdaten (80 %)
# - X_tst, y_tst: Testdaten (20 %)
# - random_state sorgt für Reproduzierbarkeit
# - stratify=y sorgt für gleiche Klassenverteilung in Train und Test
X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Eine zufällige Aufteilung ist essenziell, da eine sequentielle Teilung (z. B. erste 80 % für Training)
# systematische Verzerrungen enthalten kann (z. B. bestimmte Klassen nur am Anfang)
#
# Die Verwendung von stratify=y stellt sicher, dass alle Klassen im selben Verhältnis
# in Trainings- und Testdaten enthalten sind
#
# test_size=0.2 → 20 % der Daten werden für das Testset reserviert
# random_state=42 garantiert Reproduzierbarkeit der zufälligen Ziehung
#
# Diese Vorgehensweise erlaubt eine realistische Beurteilung der Modellgüte auf unbekannten Daten


# Importieren des DecisionTreeClassifier aus scikit-learn
# Diese Klasse implementiert den CART-Algorithmus zur Klassifikation mittels Entscheidungsbaum
from sklearn.tree import DecisionTreeClassifier

# Erstellen eines Entscheidungsbaum-Objekts
# - Standardmäßig wird das Kriterium "gini" verwendet (Gini-Unreinheit)
# - Alternative wäre z. B. 'entropy' für die ID3-ähnliche Variante
model = DecisionTreeClassifier()  # alternativ: DecisionTreeClassifier(criterion='entropy')

# Trainieren des Modells mit den Trainingsdaten
# - X_tr: Merkmale der Trainingsdaten (n_samples × n_features)
# - y_tr: Klassenlabels der Trainingsdaten (n_samples)
# Die Methode .fit() baut den Entscheidungsbaum rekursiv auf:
# - wählt pro Knoten das optimale Feature zur Aufteilung
# - verwendet dabei ein Gütemaß (Gini oder Entropie)
# - stoppt, wenn Tiefe oder Datenmenge zu gering wird
model.fit(X_tr, y_tr)

# Der DecisionTreeClassifier basiert auf dem CART-Algorithmus:
# - "Greedy"-Verfahren: trifft lokal optimale Entscheidungen beim Splitten
# - verwendet standardmäßig den Gini-Index zur Messung der Reinheit

# Alternative Kriterien:
# - criterion='entropy': verwendet Informationsgewinn, ähnlich ID3
#   geeignet für kategoriale Merkmale, erzeugt aber oft tiefere Bäume

# Wichtige Parameter zur Steuerung der Baumkomplexität:
# - max_depth: maximale Tiefe des Baumes
# - min_samples_split: minimale Anzahl von Beispielen, um einen Knoten zu teilen
# - min_samples_leaf: minimale Anzahl von Beispielen in einem Blatt
# - max_features: maximale Anzahl der Merkmale, die beim Split berücksichtigt werden
# - ccp_alpha: Kostenkomplexitäts-Pruning-Parameter (zur Vermeidung von Overfitting)

# Beispiel für Pruning:
# model = DecisionTreeClassifier(ccp_alpha=0.05)

# Wichtig:
# - .fit(X, y) darf nur auf den Trainingsdaten aufgerufen werden!
# - Ein mehrfaches Aufrufen mit verschiedenen Daten führt zu Overfitting

# --- 1. Import der Evaluierungsfunktionen ---
# Zur Beurteilung des trainierten Modells verwenden wir Metriken aus scikit-learn
from sklearn.metrics import confusion_matrix, classification_report


# --- 2. Vorhersage der Testdaten ---
# Das Modell ist bisher nur auf den Trainingsdaten X_tr, y_tr trainiert worden.
# Nun wenden wir es auf neue, ungesehene Daten an, um seine Generalisierbarkeit zu prüfen.
# Wichtig:
# - Nur auf Daten anwenden, die NICHT zum Training verwendet wurden
# - Sonst ist das Ergebnis nicht aussagekräftig zur Modellgüte auf neuen Daten

# predict() nimmt als Input die Merkmalsmatrix X_tst (Form: [n_samples, n_features])
# und gibt ein 1D-Array mit den vorhergesagten Klassenlabels zurück.
y_predicted = model.predict(X_tst)


# --- 3. Konfusionsmatrix ---
# Die confusion_matrix vergleicht wahre Klassenlabels (y_tst)
# mit den vom Modell vorhergesagten Labels (y_predicted).
# Die Matrix ist quadratisch: (n_classes × n_classes)
# Zeilen: wahre Klassen
# Spalten: vorhergesagte Klassen
# Diagonalelemente: korrekt klassifizierte Instanzen
# Off-Diagonalelemente: Fehlklassifikationen
matrix = confusion_matrix(y_tst, y_predicted)

# Matrix anzeigen
print("Confusion Matrix:")
print(matrix)

# Interpretation:
# - Je höher die Diagonaleinträge im Vergleich zu den Nebendiagonalen, desto besser
# - Analyse zeigt, welche Klassen verwechselt werden
# - Besonders wichtig bei Klassen mit ähnlicher Ausprägung oder bei Imbalancen


# --- 4. Klassifikationsbericht ---
# classification_report gibt eine Übersicht über die wichtigsten Metriken:
# - precision: Anteil der korrekt vorhergesagten Positiven
#   = TP / (TP + FP)
# - recall (Sensitivität): Anteil der erkannten Positiven unter allen tatsächlichen Positiven
#   = TP / (TP + FN)
# - f1-score: harmonisches Mittel aus precision und recall
# - support: Anzahl der wahren Instanzen pro Klasse

# Der Parameter digits=3 sorgt für drei Nachkommastellen bei der Ausgabe
report = classification_report(y_tst, y_predicted, digits=3)

# Bericht ausgeben
print("Classification Report:")
print(report)


# --- 5. Detaillierte Erläuterungen zu den Metriken ---

# Precision:
# Gibt an, wie zuverlässig positive Vorhersagen sind.
# Wichtig, wenn falsch-positive Ergebnisse vermieden werden sollen.
# Beispiele:
# - Spamfilter (falsch-positive = wichtige Mails fälschlich als Spam)
# - medizinische Tests (gesunde Patienten fälschlich als krank)
# - Betrugserkennung (legitime Transaktionen blockiert)

# Recall:
# Gibt an, wie viele der tatsächlichen Positiven erkannt wurden.
# Wichtig, wenn falsch-negative Ergebnisse vermieden werden sollen.
# Beispiele:
# - Cybersicherheit (unerkannter Angriff = großes Risiko)
# - Krankheitsdiagnose (kranker Patient nicht erkannt)
# - Verbrechensaufdeckung (unerkannte Straftat)

# F1-Score:
# Kombiniert precision und recall → Kompromissmaß
# Sinnvoll bei:
# - Imbalancierten Klassen
# - Zielkonflikten zwischen precision und recall

# macro avg:
# Ungewichteter Durchschnitt der Metriken über alle Klassen
# weighted avg:
# Durchschnitt über Klassen unter Berücksichtigung des supports (Klassenhäufigkeit)

# Gesamtziel:
# - hohe Werte bei precision, recall und f1-score → gutes, ausgewogenes Modell
# - Einseitige Schwächen (z. B. recall niedrig bei einer Klasse) → gezielte Nachjustierung nötig


# --- 1. Import der benötigten Visualisierungsbibliotheken ---
from sklearn.tree import plot_tree      # Visualisiert den Entscheidungsbaum
import matplotlib.pyplot as plt         # Für das Plotten der Grafik


# --- 2. Erzeugen einer neuen Figure für den Plot ---
# Mit plt.figure() wird ein neuer Grafikbereich erzeugt
# Der Parameter figsize steuert die Abmessung in Zoll: (Breite, Höhe)
# Hier: 12 Zoll breit, 8 Zoll hoch → gut geeignet für größere Bäume
plt.figure(figsize=(12, 8))


# --- 3. Visualisierung des Entscheidungsbaums ---
# plot_tree() erzeugt eine visuelle Darstellung des gesamten Entscheidungsbaums

# Parameter:
# - model: das trainierte DecisionTreeClassifier-Modell
# - feature_names: Liste mit Namen der Eingabemerkmale (X-Spaltennamen)
# - class_names: Klassenbezeichner für die Zielvariable
# - filled=True: Knoten werden eingefärbt nach dominierender Klasse
# - rounded=True: Knoten haben abgerundete Ecken → bessere Lesbarkeit
# - fontsize=6: Schriftgröße innerhalb der Baumknoten
plot_tree(
    model,
    feature_names=features,
    class_names=model.classes_,  # Holt die Klassen automatisch aus dem Modell
    filled=True,
    rounded=True,
    fontsize=6
)


# --- 4. Anzeigen der Grafik ---
# plt.show() öffnet ein neues Fenster (oder zeigt inline in Jupyter die Grafik)
# Ohne plt.show() wird der Baum zwar intern erstellt, aber nicht sichtbar gemacht
plt.show()

# --- Optional: Plot als Datei speichern (nach plt.show()) ---
# Das Format wird automatisch aus dem Dateinamen abgeleitet
# Beispiel: PNG für Präsentationen, PDF für Publikationen
plt.savefig("baumstruktur.png", dpi=300, bbox_inches='tight')

# Parameter:
# - dpi=300: hohe Auflösung, gut für Druck
# - bbox_inches='tight': reduziert unnötigen Rand

# --- Zusatzhinweise zur Interpretation und Verwendung ---

# Die Darstellung hilft, das Entscheidungsverhalten des Modells nachzuvollziehen:
# - Jeder Knoten zeigt die Aufspaltung basierend auf einem Merkmal (feature)
# - "gini" zeigt die Unreinheit im Knoten (je näher an 0, desto reiner)
# - "samples": Anzahl der Trainingsdaten im Knoten
# - "value": Verteilung der Klassen im Knoten
# - Farbe: stärker gefärbte Knoten = klar dominierende Klasse

# Einsatzmöglichkeiten:
# - Modellinterpretation (z. B. bei erklärungsbedürftigen Anwendungen)
# - Fehleranalyse (z. B. bei Überanpassung durch zu tiefe Bäume)
# - Kommunikation in Präsentationen oder Berichten

# Alternative Ausgabe:
# Falls du eine PDF-Datei des Baums erzeugen willst, kannst du auch
# export_graphviz() mit graphviz verwenden (z. B. für komplexere Bäume).
# Dies ist besonders nützlich bei tiefen Bäumen, die mit plot_tree() schwer lesbar sind.