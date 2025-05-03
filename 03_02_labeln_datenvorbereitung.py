import pandas as pd
import numpy as np
import keras as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

white = pd.read_csv("data/processed/winequality-white.csv", sep=";")
red = pd.read_csv("data/processed/winequality-red.csv", sep=";")

# Labeln des Datensatzes
red['label'] = 1
white['label'] = 0
wines = pd.concat([red, white], ignore_index=True)

# Rot- und Weißweindatensätze werden für Supervised Learning vorbereitet
# Dazu erhalten die Rotweindaten das Label 1, die Weißweindaten das Label 0
red['label'] = 1
white['label'] = 0

# Die beiden Datensätze werden zeilenweise zusammengeführt
# ignore_index=True sorgt dafür, dass die Indizes neu durchgezählt werden
wines = pd.concat([red, white], ignore_index=True)

# Die ersten 11 Spalten (chemische Eigenschaften) dienen als Input-Variablen,
# auch Features genannt; fixed acidity bis alcohol, also nicht die 11te Spalte
x = wines.iloc[:, 0:11]

# Die Zielvariable ist das Label (1 = Rotwein, 0 = Weißwein),
# in 1D-Array umgewandelt
y = np.ravel(wines['label'])

# Aufteilung des gesamten Datensatzes in Trainings- und Testdaten
# 70 % werden für das Training verwendet, 30 % für das spätere Testen
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Skalieren der Daten mit StandardScaler
# Unterschiedliche Skalen der Inputvariablen (z. B. pH vs. Alkohol) können
# das Training stören -- daher werden die Trainingsdaten standardisiert
# (Mittelwert 0, Varianz 1);

scaler = StandardScaler().fit(x_train)# Nur am Trainingsset fitten!
x_train = scaler.transform(x_train)   # Transformation des Trainingssets
x_test = scaler.transform(x_test)     # Gleiche Transformation für Testdaten


# Klassifikation von Weinen (rot vs. weiß) als binäres Klassifikationsproblem
# Ziel: Vorhersage d. binären Labels (0 = rot, 1 = weiß) anhand chemischer
# Eigenschaften

# Das neuronale Netz wird an diese Problemstruktur angepasst:
# - ein einzelnes Ausgabeneuron mit Sigmoid-Aktivierung (für binäre Klassifikation)
# - Eingabedimension entspricht der Anzahl der Input-Features (hier: 11)

# Keras verwendet ein sequenzielles Modell, in dem Layer nacheinander
# hinzugefügt werden. Hyperparameter wie Anzahl der Hidden-Layer,
# Neuronenanzahl, Aktivierungsfunktionen etc. sind frei wählbar


# Erstellen eines sequenziellen Keras-Modells
model = K.models.Sequential()

# Erste Schicht (Input-Layer + erste Dense-Schicht)
# - 12 Neuronen als Startwert (guter Richtwert, entspricht etwa der Feature-Anzahl)
# - Aktivierungsfunktion: ReLU (Standard bei Hidden-Layern)
# - input_dim: 11, da 11 Input-Features (chemische Eigenschaften)
model.add(K.layers.Dense(units=12, activation='relu', input_dim=11))

# Zweite Schicht (Hidden-Layer)
# - 8 Neuronen (leichte Reduktion gegenüber erster Schicht)
# - erneut ReLU-Aktivierung
model.add(K.layers.Dense(units=8, activation='relu'))

# Ausgabeschicht für binäre Klassifikation
# - 1 Neuron (0 = rot, 1 = weiß)
# - Aktivierungsfunktion: Sigmoid (liefert Werte zwischen 0 und 1 → geeignet für binäre Klassen)
model.add(K.layers.Dense(units=1, activation='sigmoid'))

# Kompilieren des Modells: Festlegen von Optimierer, Verlustfunktion und Metriken
# - Optimierer: Adam (state-of-the-art für viele Probleme, adaptiv)
# - Verlustfunktion: binary_crossentropy (geeignet für binäre Klassifikation)
# - Metrik: accuracy (Anteil korrekt klassifizierter Beispiele pro Epoche)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Das neuronale Netz ist vollständig aufgebaut:
# - Eingabeschicht mit 12 Neuronen (input_dim = 11 Merkmale)
# - 1 versteckte Schicht (Hidden Layer) mit 8 Neuronen
# - Ausgabeschicht mit 1 Neuron (Sigmoid → Wahrscheinlichkeit für Klasse 1)
# Für binäre Klassifikation genügt ein einzelnes Ausgabeneuron mit
# Sigmoid-Aktivierung, das eine Wahrscheinlichkeit im Intervall [0,1] liefert.

# Für mehrklassige Klassifikation würde man stattdessen:
# - mehrere Ausgabeneuronen (entsprechend der Klassenzahl)
# - und eine 'softmax'-Aktivierung verwenden

# Training des Modells mit Keras
# - x_train, y_train sind die vorbereiteten Trainingsdaten
# - epochs=20: 20 Trainingsdurchläufe (kann bei Bedarf erhöht werden)
# - validation_split=0.3: 30 % der Trainingsdaten werden zur Validierung abgezweigt
#   (Achtung: dies betrifft nur x_train/y_train, nicht die vorab separat gehaltenen Testdaten!)
hist = model.fit(x_train, y_train, epochs=20, validation_split=0.3)

# Visualisierung des Trainingsverlaufs
# Hinweis: 'val_accuracy' und 'accuracy' ab Keras 2.x (früher: 'val_acc', 'acc')

plt.plot(hist.history['accuracy'], label='Train Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoche')
plt.ylabel('Genauigkeit')
plt.legend()
plt.tight_layout()
plt.savefig("training_accuracy.png")  # optional speichern
plt.show()


# Bewertung des trainierten Netzes auf den echten Testdaten (x_test, y_test)
# Gibt den Loss und die Genauigkeit auf der unbekannten Testmenge zurück
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Testverlust: {test_loss:.4f}")
print(f"Testgenauigkeit: {test_acc:.4f}")

# Hinweis:
# Die Validierungsdaten während des Trainings stammen aus dem Trainingsset (validation_split)
# Die endgültige Testgenauigkeit basiert jedoch auf echten, vorher vollständig ungenutzten Testdaten
# Nur so erhält man eine realistische Schätzung der Modellgüte auf neuen Daten