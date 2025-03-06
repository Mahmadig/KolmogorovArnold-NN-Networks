#Boubaker Implementation and ChebyKAN Implementation

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import torch
from sklearn.metrics import mean_squared_error
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
X_train, X_test, y_train, y_test= X.iloc[31240:, :],X.iloc[:31240, :],y.iloc[31240:, :],y.iloc[:31240, :]
scaler=StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

def boubaker(n, x):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x
    else:
        return (2 * n - 1) * x * boubaker(n - 1, x) - (n - 1) * boubaker(n - 2, x)

class BoubakerKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(BoubakerKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.boubaker_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.boubaker_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Normalize x to [0, 1] using sigmoid
        x = torch.sigmoid(x)

        # Compute the Boubaker basis functions
        boubaker_basis = []
        for n in range(self.degree + 1):
            boubaker_basis.append(boubaker(n, x))
        boubaker_basis = torch.stack(boubaker_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Boubaker interpolation
        y = torch.einsum("bid,iod->bo", boubaker_basis, self.boubaker_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y

class BoubakerKANModel(nn.Module):
    def __init__(self, input_dim, output_dim, degree, hidden_units):
        super(BoubakerKANModel, self).__init__()
        self.layer1 = BoubakerKANLayer(input_dim, hidden_units, degree)
        self.layer2 = BoubakerKANLayer(hidden_units, hidden_units, degree)
        self.layer3 = BoubakerKANLayer(hidden_units, hidden_units, degree)
        self.layer4 = BoubakerKANLayer(hidden_units, output_dim, degree)
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

if __name__ == "__main__":
    input_dim = 6 
    output_dim = 23  
    degree = 3  
    hidden_units = 128 
    num_epochs = 1000  



    x_train_tensor = torch.Tensor(pd.DataFrame(X_train).values)
    y_train_tensor = torch.Tensor(pd.DataFrame(y_train).values)
    x_test_tensor = torch.Tensor(pd.DataFrame(X_test).values)
    y_test_tensor = torch.Tensor(pd.DataFrame(y_test).values)
    train_data = x_train_tensor
    train_labels = y_train_tensor


    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    x_test_tensor = x_test_tensor.to(device)

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

    model = BoubakerKANModel(input_dim, output_dim, degree, hidden_units)
    model.to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, criterion, optimizer, num_epochs)
    with torch.no_grad():
        pred= model(x_test_tensor)
    prd = pred.cpu()
    tst= y_test_tensor.cpu()
    print("the RMSE is : ", np.sqrt(mean_squared_error(prd,tst )))

i=128
class MNISTChebyKAN(nn.Module):
    def __init__(self):
        super(MNISTChebyKAN, self).__init__()
        self.chebykan1 = ChebyKANLayer(6, i, 4)
        self.ln1 = nn.LayerNorm(i) 
        self.chebykan2 = ChebyKANLayer(i, i, 4)
        self.ln2 = nn.LayerNorm(i)
        self.chebykan3 = ChebyKANLayer(i, 23, 4)

    def forward(self, x):
        x = x.view(-1, 6) 
        x = self.chebykan1(x)
        x = self.ln1(x)
        x = self.chebykan2(x)
        x = self.ln2(x)
        x = self.chebykan3(x)
        return x
    model = MNISTChebyKAN()
    model.to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, criterion, optimizer, num_epochs)
    with torch.no_grad():
        pred= model(x_test_tensor)
    prd = pred.cpu()
    tst= y_test_tensor.cpu()
    print("the RMSE is : ", np.sqrt(mean_squared_error(prd,tst )))
