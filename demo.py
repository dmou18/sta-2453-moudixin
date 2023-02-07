import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import KFold
from sklearn import preprocessing
from torch.nn.functional import normalize



class PutNet(nn.Module):
    """
    Example of a Neural Network that could be trained price a put option.
    TODO: modify me!
    """

    def __init__(self) -> None:
        super(PutNet, self).__init__()

        self.l1 = nn.Linear(5, 20)
        self.l2 = nn.Linear(20, 20)
        self.l3 = nn.Linear(20, 20)
        self.l4 = nn.Linear(20, 20)
        self.l5 = nn.Linear(20, 20)
        self.l6 = nn.Linear(20, 20)
        self.l7 = nn.Linear(20, 20)
        self.l8 = nn.Linear(20, 20)
        self.l9 = nn.Linear(20, 20)
        self.out = nn.Linear(20, 1)
        

    def forward(self, x):
        #x = normalize(x, dim=1, p=2)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        x = F.relu(self.l7(x))
        x = F.relu(self.l8(x))
        x = F.relu(self.l9(x))
        x = self.out(x)
        return x


def main():
    """Train the model and save the checkpoint"""

    k_folds = 20
    num_epochs = 500

    # Create model
    model = PutNet()

    # Load dataset
    df = pd.read_csv("bs-put-1k.csv")

    # Power Transform
    pt = PowerTransformer()
    pt.fit(df[["S", "K", "T", "r", "sigma"]])   
    pt_x = pt.transform(df[["S", "K", "T", "r", "sigma"]])
    
    # Set up training
    x = torch.Tensor(pt_x)
    y = torch.Tensor(df[["value"]].to_numpy())

    criterion = nn.MSELoss()
    #optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train for num_epochs epochs
    for i in range(num_epochs):
        kfold = KFold(n_splits=k_folds, shuffle=True)

        for fold,(tr_idx, val_idx) in enumerate(kfold.split(x)):
            part_y_hat = model(x[tr_idx])
            part_y = y[tr_idx]

            # Calculate training loss
            training_loss = criterion(part_y_hat, part_y)

            # Take a step
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

            # Check validation loss
            with torch.no_grad():
                validation_loss = criterion(model(x[val_idx]), y[val_idx])

            print(f"Iteration: {i} | Fold: {fold}| Training Loss: {training_loss:.4f} | Validation Loss: {validation_loss:.4f} ")

    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
