import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Utils():
    
    @staticmethod
    def get_distance(model1, model2):
        with torch.no_grad():
            model1_flattened = nn.utils.parameters_to_vector(model1.parameters())
            model2_flattened = nn.utils.parameters_to_vector(model2.parameters())
            distance = torch.square(torch.norm(model1_flattened - model2_flattened))
        return distance

    @staticmethod
    def get_distances_from_current_model(current_model, party_models):
        # Ensure both models are on the same device
        current_model = current_model.to("cuda")
        party_models = party_models.to("cuda")
        
        num_updates = len(party_models)
        distances = np.zeros(num_updates)
        for i in range(num_updates):
            distances[i] = Utils.get_distance(current_model, party_models[i])
        return distances

    def evaluate(testloader, model):
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)  # Move data and targets to the device
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                all_labels.append(target.cpu())
                all_predictions.append(predicted.cpu())
                
        all_labels = torch.cat(all_labels).numpy()        # Convert to NumPy array
        all_predictions = torch.cat(all_predictions).numpy()  # Convert to NumPy array
        return 100 * correct / total, all_labels, all_predictions

