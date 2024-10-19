import torch

class Recommender:
    """
    The Recommender class is responsible for making recommendations for users.
    """
    def __init__(self, model, num_items):
        self.model = model
        self.num_items = num_items
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def recommend_items(self, user_id, num_recommendations=5):
        """
        Recommends top-N items for the specified user.
        """
        user_id_tensor = torch.tensor([user_id]).to(self.device)
        item_ids = torch.arange(0, self.num_items).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(user_id_tensor.expand_as(item_ids), item_ids)

        top_items = torch.argsort(predictions, descending=True)[:num_recommendations]
        return top_items.cpu().numpy()
