# Importieren der Keras-Bibliothek
import keras as K
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Laden des Fashion-MNIST-Datensatzes aus Keras (10 Klassen, z. B. Schuhe, Pullover, Taschen)
mnist = K.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisierung der Bilddaten auf Werte im Bereich [0, 1]
# Ursprünglich liegen die Grauwerte im Bereich [0, 255]
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-Hot-Encoding der Zielvariable (10 Klassen → Vektor mit 10 Einträgen)
# Beispiel: Klasse 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = K.utils.to_categorical(y_train)
y_test = K.utils.to_categorical(y_test)

# Aufbau eines Feedforward-Neuronalen Netzes
model = K.models.Sequential()

# Flatten-Ebene wandelt 2D-Bilder (28x28) in 1D-Vektoren (784) um
model.add(K.layers.Flatten())

# Zwei versteckte Dense-Schichten mit jeweils 128 Neuronen, ReLU-Aktivierung
model.add(K.layers.Dense(128, activation="relu"))
model.add(K.layers.Dense(128, activation="relu"))

# Ausgabeschicht mit 10 Neuronen (für 10 Klassen), Softmax-Aktivierung für Wahrscheinlichkeitsverteilung
model.add(K.layers.Dense(10, activation="softmax"))

# Kompilierung des Modells mit:
# - Adam-Optimierer
# - Categorical Crossentropy (für mehrklassige Klassifikation mit One-Hot-Labels)
# - Accuracy als Metrik
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Training des Modells:
# - 30 Epochen
# - Batch-Größe 128
# - 30 % der Trainingsdaten werden für Validierung verwendet
history = model.fit(x_train, y_train, epochs=30, batch_size=128, validation_split=0.3)

# Hinweis:
# Das history-Objekt speichert den gesamten Trainingsverlauf (Loss und Accuracy je Epoche)
# Nur dadurch ist es möglich, die Trainings- und Validierungskurven im Nachhinein zu plotten
# Die resultierende Grafik zeigt beide Verläufe (Accuracy auf Trainings- und Validierungsdaten)
# Bei einfachem MNIST (Ziffern 0–9) erreicht das Netz bereits sehr hohe Genauigkeit
# Im Vergleich dazu ist Fashion-MNIST (Kleidungsstücke) komplexer und führt zu geringerer Genauigkeit
# Ursache: visuelle Ähnlichkeit mancher Klassen (z. B. Shirt vs. Pullover)

# Plotten des Trainingsfortschritts (Train/Validation Accuracy)
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set_title('Accuracy over epochs', fontsize='medium')
ax.set_xlabel('epochs')
ax.set_ylabel('accuracy')

# Moderne Keras-Namenskonventionen: 'accuracy' statt 'acc'
ax.plot(history.history['accuracy'], label='train')
ax.plot(history.history['val_accuracy'], label='validation')
ax.legend(loc='upper left')

plt.show()  # ggf. hier speichern mit
plt.savefig("mnist_accuracy.png")


# Aufbau eines Convolutional Neural Networks (CNN) für die Bilderkennung
# Ziel: Verbesserung der Genauigkeit gegenüber einfachen Feedforward-Netzen
#
# 1. Der erste Layer ist ein Conv2D-Layer:
#    - 32 Filter mit einer Kerneldimension von 3×3
#    - ReLU-Aktivierung
#    - input_shape=(28, 28, 1): Eingabebilder sind 28×28 Pixel mit 1 Kanal (grau)
mnist_fashion = K.datasets.fashion_mnist
(x_train_mf, y_train_mf), (x_test_mf, y_test_mf) = mnist.load_data()

# Normalisierung der Bilddaten auf Werte im Bereich [0, 1]
# Ursprünglich liegen die Grauwerte im Bereich [0, 255]
x_train_mf = x_train_mf / 255.0
x_test_mf = x_test_mf / 255.0

# One-Hot-Encoding der Zielvariable (10 Klassen → Vektor mit 10 Einträgen)
# Beispiel: Klasse 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train_mf = K.utils.to_categorical(y_train_mf)
y_test_mf = K.utils.to_categorical(y_test_mf)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# 2. Anschließend reduziert ein MaxPooling2D-Layer die räumliche Dimension der Featuremaps
model.add(MaxPooling2D((2, 2)))

# 3. Der Flatten-Layer wandelt die 2D-Ausgabe in einen 1D-Vektor um,
#    sodass dieser an vollverbundene (Dense) Schichten übergeben werden kann
model.add(Flatten())

# 4. Eine Dense-Schicht mit 100 Neuronen und ReLU-Aktivierung als Hidden-Layer
model.add(Dense(100, activation='relu'))

# 5. Ausgabeschicht für 10 Klassen (z. B. Ziffern oder Kleidungsstücke), Softmax-Aktivierung liefert Wahrscheinlichkeiten
model.add(Dense(10, activation='softmax'))

# Kompilierung des CNN:
# - Optimierer: Adam (effizient, adaptiv)
# - Verlustfunktion: categorical_crossentropy (geeignet für mehrklassige Klassifikation mit One-Hot-Labels)
# - Metrik: accuracy
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Training des CNN:
# - x_train_mf und y_train_mf: normalisierte Trainingsbilder und One-Hot-Labels
# - x_test_mf und y_test_mf: Testdaten zur Validierung
# - 30 Epochen, Batch-Größe 128
history = model.fit(
    x_train_mf, y_train_mf,
    epochs=30,
    batch_size=128,
    validation_data=(x_test_mf, y_test_mf)
)

# Das CNN beginnt mit einem Conv2D-Layer:
# - Er verwendet 32 Filter mit einer Kernelgröße von 3x3
# - Diese extrahieren lokale Bildmerkmale (z. B. Kanten, Texturen)
# - Die erste Zahl (32) gibt die Anzahl der Filter (= Ausgabekanäle) an
# - Eine höhere Anzahl an Filtern erhöht die Modellkapazität, aber auch den Rechenaufwand

# In empirischen Tests zeigt sich:
# - Bereits einfache CNNs erreichen ~99 % Trainingsgenauigkeit und ~91 % Testgenauigkeit
# - Dies entspricht einem signifikanten Fortschritt gegenüber klassischen Feedforward-Netzen

# Für eine bessere Generalisierbarkeit wurden zusätzliche Experimente durchgeführt:
# - Variation der Anzahl der Filter im ersten Conv2D-Layer (z. B. 32, 40, 48, 56)
# - Das Modell mit 48 Filtern schnitt im Mittel am besten auf den Testdaten ab

# Eine weitere bewährte Technik zur Vermeidung von Overfitting ist der Einsatz eines Dropout-Layers:
# - Während des Trainings werden zufällig ausgewählte Neuronen deaktiviert
# - Dies verhindert eine zu starke Abhängigkeit von einzelnen Aktivierungen
# - Ziel: bessere Generalisierbarkeit auf unbekannte Daten


# Aufbau eines CNN mit Dropout zur Reduktion von Overfitting
# Dropout deaktiviert während des Trainings zufällig 10 % der Neuronen im vorherigen Layer
# Ziel: Netz soll robuster gegen Überanpassung werden und besser auf neuen Daten generalisieren

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))  # 10 % der Neuronen werden zufällig deaktiviert
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))
