import torch
import torch.optim as optim
import torch.nn as nn

class Trainer:
    """
    The Trainer class handles the training and validation of the recommendation model.
    """
    def __init__(self, model, train_loader, val_loader, lr=0.01, num_epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train(self):
        """
        Train the model for a specified number of epochs.
        """
        for epoch in range(self.num_epochs):
            self.model.train()  # Set the model to training mode
            total_loss = 0

            for user, item, rating in self.train_loader:
                # Convert user and item indices to torch.long (required for embedding layers)
                user = user.to(self.device).long()
                item = item.to(self.device).long()
                rating = rating.float().to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                predictions = self.model(user, item)
                loss = self.criterion(predictions, rating)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {total_loss / len(self.train_loader)}')

    def validate(self):
        """
        Validate the model on the validation dataset and print the validation loss.
        """
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for user, item, rating in self.val_loader:
                # Convert user and item indices to torch.long (required for embedding layers)
                user = user.to(self.device).long()
                item = item.to(self.device).long()
                rating = rating.float().to(self.device)
                predictions = self.model(user, item)
                loss = self.criterion(predictions, rating)
                val_loss += loss.item()

        print(f'Validation Loss: {val_loss / len(self.val_loader)}')
