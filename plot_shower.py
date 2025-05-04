#%%
import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
# Pfad zur HDF5-Datei
file_path = "/data/dust/user/mmozzani/dataset/all_interactions_pions_regular_ECAL+HCAL_10-90GeV_25.slcio.root_with_time.hdf5"

try:
    with h5py.File(file_path, "r") as f:
        print(" Datei erfolgreich geöffnet!")

        # Datensätze laden
        energy = f["energy"][:]
        events = f["events"][:]  # shape: (19422, 5, 5000)
        n_points = f["n_points"][:]
        n_points_ecal = f["n_points_ecal"][:]
        n_points_hcal = f["n_points_hcal"][:]
        shower_start = f["shower_start"][:]

        #print("📦 Alle Datensätze erfolgreich geladen:")
        #print(f"events.shape = {events.shape}")
        #print(f"n_points.shape = {n_points.shape}")
        candidates = np.where(n_points > 4000)[0]
        print(candidates, 'iiiii')
        #print(len(candidates))
        # Wähle ein Event aus
        i = 414  # Index des Events
        n = int(n_points[i])
        event = events[i, :, :n]  # (5, n) → Features x,y,z,e,t

        # Entpacke die Features
        x, y, z, e, t = event

        #print(f"✅ Event {i} geladen mit {n} Punkten.")
        #print(f"x.shape = {x.shape}, y.shape = {y.shape}, z.shape = {z.shape}")

except Exception as e:
    print("Fehler beim Laden der Datei:")
    print(e)
    exit()

#%%
t1 = []
print(e.max())
for i in range(0, 19422):
    x, y, z, e, t = events[i]

    n = 0
    while n < len(e) and e[n] <= 0.0005:
        n += 1

    if n < len(t):
        t1.append(t[n])
#%%
plt.figure(figsize=(8, 6))
plt.hist(events[:,4][events[:,3]>0], bins=100,log=True)
plt.title("Histogram of Hit Times for Hits with Energy > 0 (MeV)")
plt.xlabel("Zeit [ns]")
#plt.xlim(, np.max(t1))
#plt.ylim(0, 600)
plt.show()

#%%
#  Plotten: y-z Projektion mit Energie als Farbe
plt.figure(figsize=(8, 6))
plt.scatter(y, z, c=e, cmap="hot", s=5)
plt.colorbar(label="Energie")
plt.xlim(np.min(y), np.max(y))
plt.ylim(np.min(z), np.max(z))

plt.xlabel("y")
plt.ylabel("z (Layer)")
plt.title(f"Pion Shower  {i} (y-z Projektion)")
plt.grid(True)
plt.tight_layout()



#  Plot speichern
output_file = f"zzz shower_event_{i} y-z plane.png"
plt.savefig(output_file)
plt.show()
#%%
plt.figure(figsize=(8, 6))
plt.scatter(z ,y, c=e, cmap="hot", s=5)
plt.colorbar(label="Energie")
plt.xlim(np.min(z), np.max(z))
plt.ylim(np.min(y), np.max(y))

plt.xlabel("z")
plt.ylabel("y (Layer)")
plt.title(f"Pion Shower  {i} (z-y Projektion)")
plt.grid(True)
plt.tight_layout()



#  Plot speichern
output_file = f"zzz shower_event_{i} z-y plane.png"
plt.savefig(output_file)

#%%

plt.figure(figsize=(8, 5))
plt.scatter(t, e, c=e, cmap="plasma", s=10)
plt.colorbar(label="Energie")
plt.xlabel("Zeit (t)")
plt.ylabel("Deponierte Energie (E)")
plt.title(f"Energie-Zeit-Verteilung von Shower Event {i}")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"zzz energy_vs_time_event_{i}.png")
#%%
plt.figure(figsize=(8, 6))
plt.scatter(x, y,  c=e, cmap="plasma", s=10)
plt.colorbar(label="Anzahl der Punkte")
plt.xlim(np.min(x), np.max(x))
plt.ylim(np.min(y), np.max(y))
plt.xlabel("x")
plt.ylabel("y") 
plt.title(f"2D Histogramm von Shower Event {i} (x-y Projektion)")
plt.grid(True)  
plt.tight_layout()
plt.savefig(f"zzz shower event{i}x, y plane.png")
#%%
plt.figure(figsize=(8, 6))
plt.scatter(x, z,  c=e, cmap="plasma", s=10) 
plt.colorbar(label="Anzahl der Punkte")
plt.xlim(np.min(x), np.max(x))
plt.ylim(np.min(z), np.max(z))
plt.xlabel("x")
plt.ylabel("z") 
plt.title(f"scatter plot{i} (x-y Projektion)")
plt.grid(True)  
plt.tight_layout()
plt.savefig(f"zzz shower event{i}x, z plane.png")

#%%
plt.figure(figsize=(8, 6))

# Normiere die Zeiten auf einen sinnvollen Größenbereich, z.B. [10, 200]
t_min, t_max = t.min(), t.max()
sizes = 10 + 190 * (t - t_min) / (t_max - t_min)

# Streu-Plot: x vs. z, Farbe = Energie, Größe = Zeit
sc = plt.scatter(x, z,
                 #c=e, cmap="plasma",
                 s=sizes, alpha=0.7,
                 edgecolors="none")

#cbar = plt.colorbar(sc)
cbar.set_label("Energie [MeV]")

plt.xlim(x.min(), x.max())
plt.ylim(z.min(), z.max())
plt.xlabel("x [mm]")
plt.ylabel("z [Layer]")
plt.title(f"Scatter plot {i} (x–z Projektion)\nPunktgröße ∝ Zeit")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"zzz_shower_event{i}_x_z_plane_time.png")
plt.show()


#%%    3D Scatter Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((2, 1, 2))
ax.view_init(elev=12, azim=-40)

# 3D­-Scatter mit logarithmischer Farbskala
sc = ax.scatter(
    x, y, z,
    c=e,
    cmap='viridis',
    norm=LogNorm(vmin=max(e.min(), 1e-6), vmax=e.max()),
    s=10,
    depthshade=True
)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Scatter of Shower Event')

# Farb­le­gen­de
cb = fig.colorbar(sc, pad=0.1, shrink=0.6)
cb.set_label('Energie pro Hit [MeV] (log scale)')

plt.tight_layout()
plt.savefig(f"zzz shower event{i}3D.png")
plt.show()
#%%

#%%
def plot_feature_hist(data, feature_name, unit, n_bins=50):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=n_bins, edgecolor='black', alpha=0.7)
    plt.xlabel(f'{feature_name} [{unit}]')
    plt.ylabel('Anzahl Hits')
    plt.title(f'Histogramm der {feature_name}-Koordinaten')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    filename = f"zzz_{feature_name}_{unit}_hist.png".replace(' ', '_')
    plt.savefig(filename)
    plt.show()
    
plot_feature_hist(x, 'x', 'mm')
plot_feature_hist(y, 'y', 'Layer')
plot_feature_hist(z, 'z', 'mm')
plot_feature_hist(e, 'energy', 'MeV')
plot_feature_hist(t, 'time', 'ns')

#%%


def plot_feature_vs_layer(layer, feature, feature_name, unit):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(layer, feature, s=10, alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel(f'{feature_name} [{unit}]')
    ax.set_title(f'{feature_name} per Layer')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fn = f"zzz_{feature_name}_vs_layer.png".replace(' ', '_')
    plt.savefig(fn)
    plt.show()

plot_feature_vs_layer(y, t, 'time', 'ns')
plot_feature_vs_layer(y, x, 'x', 'mm')
plot_feature_vs_layer(y, z, 'z', 'mm')
plot_feature_vs_layer(y, e, 'energy', 'MeV')
#%%
plt.figure(figsize=(8,4))
plt.scatter(
    np.arange(len(shower_start)),  # Event-Index
    shower_start,                  # physikalische erste Hit-Zeit in ns
    s=10, alpha=0.6
)
plt.xlabel("Index")
plt.ylabel("earliest time [ns]")
plt.title("Pearliest time per shower")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("zzz earliest_time_per_shower.png")
plt.show()
#%%
print(shower_start, min(shower_start), max(shower_start))
plt.figure(figsize=(8,4))
plt.hist(shower_start,bins=100)
n=np.linspace(0,len(shower_start),len(shower_start))
#plt.plot(shower_start)
plt.xlabel("n")            
plt.ylabel("#")
plt.title("shower_start")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
#%%
path = "/data/dust/user/mmozzani/dataset"
fn_bottom = "layer_bottom_pos_ecal+hcal.npy"
fn_top    = "layer_top_pos_ecal+hcal.npy"

# Dateien laden
bottom = np.load(f"{path}/{fn_bottom}", allow_pickle=False)
top    = np.load(f"{path}/{fn_top}",    allow_pickle=False)

# Kurz prüfen
distances = top - bottom

# 6) Ausgabe der Distanzen
print("distances:", distances)
z_pos = 0.5 * (bottom + top)  # mm

# 3) Abstand relativ zur ersten Layer
delta = z_pos - z_pos[0]      # mm

# 4) Lichtgeschwindigkeit in mm/ns
c_mm_per_ns = 299.792458      # mm pro ns

# 5) Erwartete Zeit pro Layer
t_c_full = delta / c_mm_per_ns     # ns

# 6) Layer-Indizes
layers = np.arange(len(t_c_full))

# Ergebnis ausgeben oder zurückgeben
print("layers:", layers, len(layers))
#print("t_c [ns]:", t_c, len(t_c))
#%%
#t_c_shower = t_c[:len(layers)] 

print('--------------------------------')
print('--------------------------------')
candidates = np.where(n_points > 4000)[0]
print(candidates)
for i in candidates: 
    n = int(n_points[i])
    event = events[i, :, :n]  # (5, n) → Features x,y,z,e,t
    x, y, z, e, t = event
    plt.figure(figsize=(6, 5))
    # Scatter: Layer-Index gegen Zeit
    mask_ecal = (y < 30)
    mask_hcal = (y >= 30)
    plt.scatter(y[mask_ecal], t[mask_ecal], s=2, color="blue", alpha=0.6, label="GEANT4 hits, Ecal (0-30)")
    plt.scatter(y[mask_hcal], t[mask_hcal], s=2, color="green", alpha=0.6, label="GEANT4 hits, Hcal (31-78)")
    layers      = np.unique(y)
    tmin_layer  = [t[y == lyr].min() for lyr in layers] # hier werden die min Werte für die Zeit pro Layer gespeichert
    #plt.plot(layers, tmin_layer, color="red", lw=2, label="lowest time per layer")
    layers = np.unique(y).astype(int)     # wirklich nur die Layer mit Hits
    t_c    = t_c_full[layers] 
    plt.plot(layers, t_c, color="red", lw=2, label="c‐limit")
    plt.xlabel("y-layer", fontsize=12)
    plt.ylabel("Time [ns]", fontsize=12)
    plt.xlim(0,78)
    plt.ylim(0,7)
    plt.title(f"Propagation of shower {i}", fontsize=14)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

#%%
candidates = np.where(n_points > 4000)[0]
layer=12
for i in candidates: 
    n = int(n_points[i])
    event = events[i, :, :n]  # (5, n) → Features x,y,z,e,t
    x, y, z, e, t = event
    mask = (z.astype(int) == layer)
    t_layer = t[mask]
    # nach Zeit sortieren
    idx = np.argsort(t_layer)
    t_sorted = t_layer[idx]
    hits = np.arange(len(t_sorted))

    # Perzentile berechnen
    cut50 = int(0.5 * len(t_sorted))
    cut10 = int(0.9 * len(t_sorted))

    # Plot
    plt.figure(figsize=(6, 3))
    plt.scatter(hits, t_sorted, s=20, color='gray')
    plt.axvline(cut50, color='red',   linestyle='--', label='50 % percentile')
    plt.axvline(cut10, color='blue',  linestyle='--', label='90 % percentile')
    plt.xlabel("hit")
    plt.ylabel("time [ns]")
    plt.title(f"Time in layer {layer} of one shower")
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
n_points = np.sum(events[:,0,:] != 0, axis=1) # echte Hits pro Event zählen
good_events_idx = np.where(n_points > 100)[0]
events = events[good_events_idx]
energy = energy[good_events_idx]
features_mean = np.mean(events, axis=2) 
split_idx = int(len(features_mean) * 0.8) # 80% der daten werden trainingsdaten 20% werden testdaten
# Splitten der Daten in Trainings- und Testdaten
X_train, X_test = features_mean[:split_idx], features_mean[split_idx:] # x train und testdaten werde hier erstellt
y_train, y_test = energy[:split_idx], energy[split_idx:] # y train und testdaten werden hier erstellt
X_train_tensor = torch.tensor(X_train, dtype=torch.float32) # umwandeln in Tensor
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # umwandlung in tensor unsqueeze(1) damit die dimensionen übereinstimmen 

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Output: Vorhersage der Gesamtenergie
        )

    def forward(self, x):
        return self.model(x)
model = SimpleNet()

# Loss-Funktion und Optimizer definieren
criterion = nn.MSELoss()  # Mean Squared Error für Regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ▸ Training des Modells
epochs = 30
losssafe=[]
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()          # Gradienten zurücksetzen
        outputs = model(inputs)        # Vorhersagen machen
        loss = criterion(outputs, targets)  # Loss berechnen
        loss.backward()                # Backpropagation
        optimizer.step()               # Parameter aktualisieren
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoche {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    losssafe.append(avg_loss)

# ▸ Evaluation des Modells
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor).item()
    print(f"\nTest Loss: {test_loss:.4f}")

n=np.linspace(1,30,30)
plt.figure(figsize=(8, 6))
plt.plot(n,losssafe)
plt.savefig('loss.png')
plt.figure(figsize=(8, 6))
# %%
