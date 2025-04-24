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
        #candidates = np.where(n_points > 3000)[0]
        #print(candidates, 'iiiii')
        #print(len(candidates))
        # Wähle ein Event aus
        i = 10105  # Index des Events
        n = int(n_points[i])
        event = events[i, :, :n]  # (5, n) → Features x,y,z,e,t

        # Entpacke die Features
        x, y, z, e, t = event

        #print(f"✅ Event {i} geladen mit {n} Punkten.")
        #print(f"x.shape = {x.shape}, y.shape = {y.shape}, z.shape = {z.shape}")

except Exception as e:
    print("❌ Fehler beim Laden der Datei:")
    print(e)
    exit()
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

#%%    3D Scatter Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((2, 1, 2))

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
ax.set_ylabel('Layer (y)')
ax.set_zlabel('z')
ax.set_title('3D Scatter of Shower Event')

# Farb­le­gen­de
cb = fig.colorbar(sc, pad=0.1, shrink=0.6)
cb.set_label('Energie pro Hit [MeV] (log scale)')

plt.tight_layout()
plt.savefig(f"zzz shower event{i}3D.png")
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
