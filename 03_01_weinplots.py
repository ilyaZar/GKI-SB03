import pandas as pd
import matplotlib.pyplot as plt


white = pd.read_csv("data/processed/winequality-white.csv", sep=";")
red = pd.read_csv("data/processed/winequality-red.csv", sep=";")

# Plot der Alkoholwerte der Datensätze im Vergleich
# fig das gesamte Figure-Objekt, dh. container f. beide subplots
# ax ist ein Array mit den zwei Subplot Objekten
fig, ax = plt.subplots(1, 2)
# Erzeugen eines Histogramms für den Rotwein-Datensatz
ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Rotwein")
# Erzeugen eines Histogramms für den Weißwein-Datensatz
ax[1].hist(white.alcohol, 10, facecolor='green', alpha=0.5, label="Weißwein")
# ax[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="Weißwein")
# # Positionierung der Subplots
fig.subplots_adjust(left=0.15, right=0.9, bottom=0.25, top=0.75, hspace=0.05, wspace=0.5)
# Setzen der oberen Grenze für die Häufigkeiten
ax[0].set_ylim([0, 1000])
# Beschriftung der x/y-Achse des 1. Subplots
ax[0].set_xlabel("Alkoholvolumen in %")
ax[0].set_ylabel("Häufigkeiten")
# Beschriftung der x/y-Achse des 2. Subplots
ax[1].set_xlabel("Alkoholvolumen in %")
ax[1].set_ylabel("Häufigkeiten")
# Überschrift des Gesamtplots
fig.suptitle("Verteilung nach Alkoholvolumen in %")

# Speichern statt anzeigen
plt.savefig("alkoholverteilung.png")