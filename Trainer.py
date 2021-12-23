import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Here we define the model modules
        

    def forward(self, x):
        # defines the forward function of the model. 
        raise NotImplementedError


    def fit(self, train_dataloader, test_dataloader, optimizer, epochs, device, plot_loss=False):
        train_losses = []
        test_losses = []
        for epoch in range(epochs):
            train_running_loss = self.train_epoch(
                train_dataloader=train_dataloader, 
                optimizer=optimizer, 
                epoch_idx=epoch,
                device=device)
            test_running_loss = self.predict(
                test_dataloader = test_dataloader,
                device=device)
            
            train_losses.append(train_running_loss)
            test_losses.append(test_running_loss)

        if plot_loss:
            self.plot_loss_progression(train_losses = train_losses, test_losses = test_losses)

    def plot_loss_progression(self, train_losses, test_losses):
        plt.plot(train_losses, label = "Train Loss")
        plt.plot(test_losses, label = "Test Loss")
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel("Loss")
        plt.title("Loss progression across epochs")

    def train_epoch(self, train_dataloader, optimizer, epoch_idx, device):
        running_loss = 0.0
        criterion = nn.BCEWithLogitsLoss()
        self.train()
        tk0 = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch_idx}")
        for batch_idx, (data, target) in enumerate(tk0):
            data, target = data.to(device), target.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = self(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            tk0.set_postfix(loss=avg_loss, stage="train")

        
        return running_loss / len(train_dataloader.dataset)


    def predict(self, test_dataloader, device):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)

                output = self(data)
                loss = F.cross_entropy(output, target)
                test_loss += loss.item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss /= len(test_dataloader.dataset)
        accuracy = 100. * correct / len(test_dataloader.dataset)

        return test_loss

        #print(f'Test set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(train_dataloader.dataset)} ({accuracy:.0f}%)')