
import torch 
import matplotlib.pyplot as plt

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = criterion(model(inputs), targets)
            test_loss += loss.item()
    return test_loss / len(test_loader)

# Visualization utility
def visualize_results(model, test_loader, device):
    test_batch = next(iter(test_loader))
    noisy_images, clean_images = test_batch
    noisy_images = noisy_images.to(device)
    denoised_images = model(noisy_images).detach().cpu()
    noisy_images = noisy_images.cpu()
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    for i in range(5):
        noisy_image = noisy_images[i * 2].permute(1, 2, 0)
        denoised_image = denoised_images[i * 2].permute(1, 2, 0)
        axes[0, i].imshow(noisy_image)
        axes[1, i].imshow(denoised_image)
    plt.show()
