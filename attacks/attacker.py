import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim



"""
Parameters:
    - model: a neural network model set to evaluation mode.
    - image: a tensor representing a single input image.
    - epsilon: a float that controls the intensity of the noise.

Returns:
    - adv_image: the adversarial image created by adding noise to the original.
"""
def noise_adversarial_attack(image, epsilon=0.1):

    # Generate random noise
    noise = torch.randn_like(image) * epsilon

    # Add noise to the original image
    adv_image = image + noise

    # Clip the values to maintain the orignal transformed CIFAR10 image range
    adv_image = torch.clamp(adv_image, -1.9886685609817505, 2.12648868560791) # VGG
    # adv_image = torch.clamp(adv_image, -1, 1) # ViT

    return adv_image


"""
Parameters:
    - model: model to be attacked.
    - image: tensor of the input image.
    - label: Correct label of the input image.
    - epsilon: Perturbation magnitude (noise level).

Returns:
    - perturbed_image: Adversarial image obtained after perturbation.
"""
def fgsm_adversarial_attack(model, image, target, epsilon=0.1):

    # Enable gradient tracking
    image.requires_grad = True

    output = model(image)

    model.zero_grad()

    loss = nn.CrossEntropyLoss()(output, target)
    
    loss.backward()

    # Collect the element-wise sign of the data gradient
    sign_data_grad = image.grad.sign()

    # Create the perturbed image by adjusting the original image
    perturbed_image = image + epsilon * sign_data_grad

    # Clip the values to maintain the image range
    perturbed_image = torch.clamp(perturbed_image, -1.9886685609817505, 2.12648868560791)

    return perturbed_image


"""
Parameters:
    - model: model to be attacked.
    - image: tensor of the input image.
    - label: Correct label of the input image.
    - epsilon: Perturbation magnitude (noise level).
    - alpha: Step size for the iterative method.
    - iters: Number of iterations for the iterative method.

Returns:
    - perturbed_image: Adversarial image obtained after perturbation.
"""
def pgd_adversarial_attack(model, image, target, epsilon=0.01, alpha=0.01, iters=10):

    # Create a copy of the input image
    perturbed_image = image.detach().clone()

    for i in range(iters):

        # Enable gradient tracking
        perturbed_image.requires_grad = True

        output = model(perturbed_image)

        model.zero_grad()

        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Collect the element-wise sign of the data gradient
        sign_data_grad = perturbed_image.grad.sign()

        with torch.no_grad():
            # Create the perturbed image by adjusting the original image
            perturbed_image = perturbed_image + alpha * sign_data_grad

            # Clip the values to maintain the image range
            perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
            perturbed_image = torch.clamp(perturbed_image, -1.9886685609817505, 2.12648868560791)
        
        perturbed_image = perturbed_image.detach()
    
    return perturbed_image


# Modify the function according to the CW attack algorithm package
def cw_adversarial_attack(model, images, labels, targeted=False, c=1e-4, kappa=0, max_iter=100, learning_rate=0.01, device=None):

    model.to(device)
    images = images.to(device)     
    labels = labels.to(device)

    # Define f-function
    def f(x) :

        outputs = model(x)

        one_hot_labels = torch.eye(len(outputs[0]), device=device)[labels]

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        return torch.clamp(j-i, min=-kappa)
    
    w = torch.zeros_like(images, requires_grad=True).to(device)
    optimizer = optim.Adam([w], lr=learning_rate)

    prev = float('inf')

    for step in range(max_iter) :
        
        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost
        
        print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')

    attack_images = 1/2*(nn.Tanh()(w) + 1)

    return attack_images

