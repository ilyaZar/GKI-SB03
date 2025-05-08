import matplotlib.pyplot as plt
# Plotten von Temperaturkurven
Tage = list(range(1, 9))
Grad_min = [19.6, 24.1, 26.7, 28.3, 27.5, 30.5, 32.8, 33.1]
Grad_max = [24.8, 28.9, 31.3, 33.0, 34.9, 35.6, 38.4, 39.2]
# Beschriftung der x-Achse
plt.xlabel('Tag')
# Beschriftung der y-Achse
plt.ylabel('Temperatur in Grad Celsius')
# Erzeugen des Plots für die Minimaltemperaturen
plt.plot(Tage, Grad_min)
# Hinzufügen von Punkten für die Einträge der Minimaltemperaturen
plt.plot(Tage, Grad_min, "oy")
# Erzeugen des Plots für die Maximaltemperaturen
plt.plot(Tage, Grad_max)
# Hinzufügen von Punkten für die Einträge der Maximaltemperaturen
plt.plot(Tage, Grad_max, "or")
# Festlegung der Achsendimension x: 0–10, y: 14–45
xmin, xmax, ymin, ymax = 0, 10, 14, 45
# Erzeugen der Achsen
# plt.axis([xmin, xmax, ymin, ymax])
plt.axis((xmin, xmax, ymin, ymax))
# Darstellung des Plots
plt.show()
plt.savefig("temperaturverlauf.png")