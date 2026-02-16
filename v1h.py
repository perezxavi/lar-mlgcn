import torch
import torch.nn as nn
import torch.nn.functional as F
from skmultilearn.dataset import load_dataset
import numpy as np

# ---------- Cargar el dataset ----------
print("Cargando dataset RCV1subset1...")
X_train, Y_train, _, _ = load_dataset('enron', 'train')
X_test, Y_test, _, _ = load_dataset('enron', 'test')

X_train = torch.tensor(X_train.toarray(), dtype=torch.float)
Y_train = torch.tensor(Y_train.toarray(), dtype=torch.float)
X_test = torch.tensor(X_test.toarray(), dtype=torch.float)
Y_test = torch.tensor(Y_test.toarray(), dtype=torch.float)

num_features = X_train.shape[1]
num_classes = Y_train.shape[1]
hidden_dim = 64

# ---------- Modelo base (MLP) ----------
class TabularEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
    def forward(self, x):
        return self.network(x)

# ---------- Capa GCN simple ----------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0))  # autoconexiones
        deg = adj.sum(dim=1, keepdim=True)
        norm_adj = adj / deg
        return F.relu(self.linear(norm_adj @ x))

# ---------- Generador de grafo dinámico ----------
class DynamicGraph(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes * num_classes)
        self.num_classes = num_classes
    def forward(self, x):
        A_flat = torch.sigmoid(self.fc(x))
        return A_flat.view(self.num_classes, self.num_classes)

# ---------- Inicializar modelos ----------
embedder = TabularEmbedder(num_features, hidden_dim)
gcn = GCNLayer(hidden_dim, hidden_dim)
dynamic_graph = DynamicGraph(num_features, num_classes)

# Embeddings de etiquetas iniciales (parámetros aprendibles)
label_embeddings = nn.Parameter(torch.randn(num_classes, hidden_dim))

# ---------- Seleccionar una instancia de prueba ----------
x_sample = X_test[0].unsqueeze(0)  # [1, num_features]
y_true = Y_test[0]

# ---------- Grafo Estático (identidad: sin relaciones explícitas) ----------
static_adj = torch.eye(num_classes)
h_i = embedder(x_sample)
ref_labels_static = gcn(label_embeddings, static_adj)
scores_static = torch.sigmoid(torch.matmul(ref_labels_static, h_i.squeeze().T))

# ---------- Grafo Dinámico generado desde x_sample ----------
adj_dynamic = dynamic_graph(x_sample)
ref_labels_dynamic = gcn(label_embeddings, adj_dynamic)
scores_dynamic = torch.sigmoid(torch.matmul(ref_labels_dynamic, h_i.squeeze().T))

# ---------- Mostrar resultados ----------
print("\nTrue labels (Y_test[0]):")
print(y_true.numpy().astype(int))

print("\nPredicted scores with STATIC graph:")
print(np.round(scores_static.detach().numpy(), 3))

print("\nPredicted scores with DYNAMIC graph:")
print(np.round(scores_dynamic.detach().numpy(), 3))
