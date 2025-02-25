from statistics import harmonic_mean
F1_scores = {'pathmnist':0.9217, 
             'dermamnist':0.7232, 
             'octmnist':0.8626, 
             'pneumoniamnist':0.8752, 
             'retinamnist':0.5071, 
             'breastmnist':0.8605, 
             'bloodmnist': 0.9847, 
             'tissuemnist': 0.5079, 
             'organamnist': 0.9467, 
             'organcmnist': 0.9318, 
             'organsmnist': 0.7733
             }

print(f"Result: {harmonic_mean(F1_scores.values()):.4f}")