import torch
from data_loader import UserItemRatingDataset
from model import MatrixFactorization
from trainer import Trainer
from recommender import Recommender
from utils import save_model, load_model

def main():
    # Load data and create DataLoaders
    train_loader, val_loader = UserItemRatingDataset.get_data_loaders('user_ratings.csv', batch_size=64)

    # Create the recommendation model
    num_users = len(train_loader.dataset.dataset.users)
    num_items = len(train_loader.dataset.dataset.items)
    model = MatrixFactorization(num_users, num_items, embedding_size=50)

    # Train the model
    trainer = Trainer(model, train_loader, val_loader, lr=0.01, num_epochs=10)
    trainer.train()
    trainer.validate()

    # Save the model
    save_model(model, 'recommendation_model.pth')

    # Load the model for recommendations
    load_model(model, 'recommendation_model.pth')

    # Make recommendations
    recommender = Recommender(model, num_items)
    recommendations = recommender.recommend_items(user_id=0, num_recommendations=5)
    print(f"Recommended items for user 0: {recommendations}")

if __name__ == "__main__":
    main()
