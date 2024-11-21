import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
#
from script.GCN import *
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#
import pickle
with open(parent_dir+'/Dataset/Bergman_cyclization.pkl', 'rb') as file:
    data_list = pickle.load(file)

for tmp in data_list:
    new_feature_x = encode_features(tmp.x)
    tmp.x = torch.tensor(new_feature_x, dtype=torch.float)

print(data_list[0].x.shape)

total_data = [get_sub_G_list(i) for i in data_list]


#
train_data, test_data = train_test_split(total_data, test_size=0.2, random_state=2024)
in_features = 134
hidden_features = 64
out_features = 1
# model = GCN_PyG(in_features, hidden_features, out_features)

model = torch.load('GCN_PyG.pth')


loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
epochs = 500
train_losses = []
test_losses = []
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for i, (subgraph_1, subgraph_2, subgraph_3, y) in enumerate(train_data):
        output = model(subgraph_1, subgraph_2, subgraph_3)
        loss = loss_fn(output, y)
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_train_loss = total_train_loss / (i + 1)
    train_losses.append(avg_train_loss)
    scheduler.step()
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for i, (subgraph_1, subgraph_2, subgraph_3, y) in enumerate(test_data):
            output = model(subgraph_1, subgraph_2, subgraph_3)
            loss = loss_fn(output, y)
            total_test_loss += loss.item()
    avg_test_loss = total_test_loss / (i + 1)
    test_losses.append(avg_test_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}")


predictions = []
actuals = []
model.eval()
with torch.no_grad():
    for i, (subgraph_1, subgraph_2, subgraph_3, y) in enumerate(total_data):
        output = model(subgraph_1, subgraph_2, subgraph_3)
        predictions.append(output.item())
        actuals.append(y.item())

plt.figure(figsize=(10, 6))
plt.scatter(actuals, predictions, alpha=0.5)
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Predictions vs Actual Values')
plt.legend()
plt.show()

from sklearn.metrics import r2_score
predictions = []
actuals = []
model.eval()
with torch.no_grad():
    for i, (subgraph_1, subgraph_2, subgraph_3, y) in enumerate(test_data):
        output = model(subgraph_1, subgraph_2, subgraph_3)
        predictions.append(output.item())
        actuals.append(y.item())

r2 = r2_score(actuals, predictions)
print(f"R²: {r2:.4f}")

torch.save(model, 'GCN_PyG2.pth')



#
