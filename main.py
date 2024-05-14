import os
import copy
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.vgg11 import VGG
from transformers import ViTForImageClassification, TrainingArguments, Trainer, ViTImageProcessor

import torchattacks
from attacks.attacker import noise_adversarial_attack, fgsm_adversarial_attack, pgd_adversarial_attack, cw_adversarial_attack
from utils import CIFAR10Data, CIFAR10ViT, compute_metrics, collate_fn, AdversarialDataset

torch.cuda.empty_cache()
torch.cuda.manual_seed(42)

######################################### VGG11 #########################################

def train_vgg(args, device):

    # Load CIFAR-10 dataset 
    data = CIFAR10Data(args)
    
    model = VGG()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, 
                                momentum=args.momentum, weight_decay=args.weight_decay)

    best_model = None
    best_accuracy = 0.0
    train_losses = []
    train_acc = []
    test_acc = []

    for epoch in range(args.num_epochs):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(data.train_dataloader()):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(data.train_dataloader().dataset),
                    100. * batch_idx / len(data.train_dataloader()), loss.item()))

        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, targets in data.test_dataloader():
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

        accuracy = correct / len(data.test_dataloader().dataset)
        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(data.test_dataloader().dataset), 100. * accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)

    # save the best model
    torch.save(best_model.state_dict(), os.path.join("./state_dicts/", 'vgg11.pth'))

def eval_vgg(args, device):

    data = CIFAR10Data(args)

    model = VGG()
    model.load_state_dict(torch.load(os.path.join("./state_dicts/vgg11/", 'vgg11_bn.pth')))
    model.to(device)

    model.eval()

    correct = 0
    with torch.no_grad():
        for inputs, targets in data.test_dataloader():
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    accuracy = correct / len(data.test_dataloader().dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(data.test_dataloader().dataset), 100. * accuracy))

" Attack functions "

def noise_adv_attack(args, device, epsilon=0.1):

    data = CIFAR10Data(args)

    model = VGG()
    model.load_state_dict(torch.load(os.path.join("./state_dicts/vgg11/", 'vgg11_bn.pth')))
    model.to(device)

    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, targets in data.test_dataloader():
            inputs, targets = inputs.to(device), targets.to(device)

            # Generate adversarial examples
            adv_inputs = noise_adversarial_attack(inputs, epsilon=epsilon)
            
            outputs = model(adv_inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    accuracy = correct / len(data.test_dataloader().dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(data.test_dataloader().dataset), 100. * accuracy))

    return accuracy

def fgsm_adv_attack(args, device, epsilon=0.01):
    
    data = CIFAR10Data(args)

    model = VGG()
    model.load_state_dict(torch.load(os.path.join("./state_dicts/vgg11/", 'vgg11_bn.pth')))
    model.to(device)

    correct = 0
    for inputs, targets in data.test_dataloader():
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Generate adversarial examples
        adv_inputs = fgsm_adversarial_attack(model, inputs, targets, epsilon=epsilon)
        
        model.eval()
        with torch.no_grad():
            outputs = model(adv_inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    accuracy = correct / len(data.test_dataloader().dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(data.test_dataloader().dataset), 100. * accuracy))

    return accuracy

def pgd_adv_attack(args, device, epsilon=0.01):
    
    data = CIFAR10Data(args)

    model = VGG()
    model.load_state_dict(torch.load(os.path.join("./state_dicts/vgg11/", 'vgg11_bn.pth')))
    model.to(device)

    correct = 0
    for inputs, targets in data.test_dataloader():
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Generate adversarial examples
        adv_inputs = pgd_adversarial_attack(model, inputs, targets, epsilon=epsilon, alpha=0.01, iters=10)
        
        model.eval()
        with torch.no_grad():
            outputs = model(adv_inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    accuracy = correct / len(data.test_dataloader().dataset)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(data.test_dataloader().dataset), 100. * accuracy))

    return accuracy

def cw_adv_attack(args, device, c=0.01):

    data = CIFAR10Data(args)

    model = VGG()
    model.load_state_dict(torch.load(os.path.join("./state_dicts/vgg11/", 'vgg11_bn.pth')))
    model.to(device)

    correct = 0
    for inputs, targets in data.test_dataloader():
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Generate adversarial examples
        # adv_inputs = cw_adversarial_attack(model, inputs, targets, c=1e-4, kappa=0.1, max_iter=500, device=device)
        attack = torchattacks.CW(model, c=c, kappa=0, steps=50, lr=0.01)
        adv_inputs = attack(inputs, targets)
        
        model.eval()
        with torch.no_grad():
            outputs = model(adv_inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    accuracy = correct / len(data.test_dataloader().dataset)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(data.test_dataloader().dataset), 100. * accuracy))

    return accuracy

" Defend functions "
#TODO Reimplement the adv_train functions

def noise_adv_train(args, device, epsilon=0.1):
    
    data = CIFAR10Data(args)

    model = VGG()
    model.load_state_dict(torch.load(os.path.join("./state_dicts/vgg11/", 'vgg11_bn.pth')))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, 
                                momentum=args.momentum, weight_decay=args.weight_decay)

    best_model = None
    best_accuracy = 0.0
    train_losses = []
    train_acc = []
    test_acc = []

    min_loss = 100000
    es = 0
    for epoch in range(args.num_epochs):
        
        total_loss = 0
        train_correct = 0
        # Perform adversarial training
        model.train()
        for batch_idx, (inputs, targets) in enumerate(data.train_dataloader()):
            
            inputs, targets = inputs.to(device), targets.to(device)
            # Generate adversarial examples
            adv_inputs = noise_adversarial_attack(inputs, epsilon=epsilon)
            
            optimizer.zero_grad()
            outputs = model(adv_inputs)

            # calculate the training acc
            _, predicted = outputs.max(1)

            loss = criterion(outputs, targets)
            loss.backward()
            
            optimizer.step()

            train_correct += predicted.eq(targets).sum().item()
            total_loss += loss.item()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(data.train_dataloader().dataset),
                    100. * batch_idx / len(data.train_dataloader()), loss.item()))
        print('Train set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_correct, len(data.train_dataloader().dataset), 100. * train_correct / len(data.train_dataloader().dataset)))
        train_loss = total_loss / len(data.train_dataloader())
        train_losses.append(train_loss)

        # Evaluate the model after each epoch
        model.eval()
        eval_correct = 0
        with torch.no_grad():
            for inputs, targets in data.test_dataloader():
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Generate adversarial examples
                adv_inputs = noise_adversarial_attack(inputs, epsilon=epsilon)
                
                outputs = model(adv_inputs)
                _, predicted = outputs.max(1)
                eval_correct += predicted.eq(targets).sum().item()

            accuracy = eval_correct / len(data.test_dataloader().dataset)
            print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
                eval_correct, len(data.test_dataloader().dataset), 100. * accuracy))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)

        # Early Stopping
        if train_loss < min_loss:
            min_loss = train_loss
            es = 0
        else:
            es += 1
            if es > 5:
                break

    # save the best model
    torch.save(best_model.state_dict(), os.path.join("./state_dicts/vgg11/", 'vgg11_noise_adv.pth'))

    return best_accuracy, train_losses

def fgsm_adv_train(args, device, epsilon=0.1):

    data = CIFAR10Data(args)

    model = VGG()
    model.load_state_dict(torch.load(os.path.join("./state_dicts/vgg11/", 'vgg11_bn.pth')))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    best_model = None
    best_accuracy = 0.0
    train_losses = []
    train_acc = []
    test_acc = []

    min_loss = 100000
    es = 0
    for epoch in range(args.num_epochs):
        
        # Perform adversarial training
        model.train()

        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(data.train_dataloader()):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Generate adversarial examples
            adv_inputs = fgsm_adversarial_attack(model, inputs, targets, epsilon=epsilon)
            
            optimizer.zero_grad()
            outputs = model(adv_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(data.train_dataloader().dataset),
                    100. * batch_idx / len(data.train_dataloader()), loss.item()))

        train_loss = total_loss / len(data.train_dataloader())
        train_losses.append(train_loss)

        # Evaluate the model after each epoch
        correct = 0
        for inputs, targets in data.test_dataloader():
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Generate adversarial examples
            adv_inputs = fgsm_adversarial_attack(model, inputs, targets, epsilon=epsilon)
            
            model.eval()
            with torch.no_grad():
                outputs = model(adv_inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

        accuracy = correct / len(data.test_dataloader().dataset)
        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(data.test_dataloader().dataset), 100. * accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)

        # Early Stopping
        if train_loss < min_loss:
            min_loss = train_loss
            es = 0
        else:
            es += 1
            if es > 5:
                break

    # save the best model
    torch.save(best_model.state_dict(), os.path.join("./state_dicts/", 'vgg11_fgsm_adv.pth'))

    return best_accuracy, train_losses

def pgd_adv_train(args, device):
        
    data = CIFAR10Data(args)

    model = VGG()
    model.load_state_dict(torch.load(os.path.join("./state_dicts/vgg11/", 'vgg11_bn.pth')))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    best_model = None
    best_accuracy = 0.0
    train_losses = []
    train_acc = []
    test_acc = []

    for epoch in range(args.num_epochs):
        
        # Perform adversarial training
        model.train()
        for batch_idx, (inputs, targets) in enumerate(data.train_dataloader()):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Generate adversarial examples
            adv_inputs = pgd_adversarial_attack(model, inputs, targets, epsilon=0.01, alpha=0.01, iters=10)
            
            optimizer.zero_grad()
            outputs = model(adv_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(data.train_dataloader().dataset),
                    100. * batch_idx / len(data.train_dataloader()), loss.item()))

        # Evaluate the model after each epoch
        correct = 0
        for inputs, targets in data.test_dataloader():
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Generate adversarial examples
            adv_inputs = pgd_adversarial_attack(model, inputs, targets, epsilon=0.01, alpha=0.01, iters=10)
            
            model.eval()
            with torch.no_grad():
                outputs = model(adv_inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

        accuracy = correct / len(data.test_dataloader().dataset)
        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(data.test_dataloader().dataset), 100. * accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)

    # save the best model
    # torch.save(best_model.state_dict(), os.path.join("./state_dicts/", 'vgg11_adv.pth'))

def cw_adv_train(args, device):
    
    data = CIFAR10Data(args)

    model = VGG()
    model.load_state_dict(torch.load(os.path.join("./state_dicts/vgg11/", 'vgg11_bn.pt')))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    best_model = None
    best_accuracy = 0.0
    train_losses = []
    train_acc = []
    test_acc = []

    for epoch in range(args.num_epochs):
        
        # Perform adversarial training
        model.train()
        for batch_idx, (inputs, targets) in enumerate(data.train_dataloader()):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Generate adversarial examples
            attack = torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01)
            adv_inputs = attack(inputs, targets)
            
            optimizer.zero_grad()
            outputs = model(adv_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(data.train_dataloader().dataset),
                    100. * batch_idx / len(data.train_dataloader()), loss.item()))

        # Evaluate the model after each epoch
        correct = 0
        for inputs, targets in data.test_dataloader():
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Generate adversarial examples
            attack = torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01)
            adv_inputs = attack(inputs, targets)

            model.eval()
            with torch.no_grad():
                outputs = model(adv_inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

        accuracy = correct / len(data.test_dataloader().dataset)
        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(data.test_dataloader().dataset), 100. * accuracy))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)

    # save the best model
    # torch.save(best_model.state_dict(), os.path.join("./state_dicts/", 'vgg11_adv.pth'))

######################################### ViT #########################################

def train_vit():

    # Load the dataset
    data = CIFAR10ViT()
    train_dataset, val_dataset, test_dataset = data.get_datasets()

    # Load the pre-trained ViT model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  id2label=data.id2label,
                                                  label2id=data.label2id)

    # Fine-tune the pre-trained model
    args = TrainingArguments(
        f"./state_dicts/test-cifar-10",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='logs',
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=data.processor,
    )
    trainer.train()

def eval_vit():
    
    # Load the test dataset
    data = CIFAR10ViT()
    train_dataset, val_dataset, test_dataset = data.get_datasets()

    # Load the fine-tuned model
    checkpoint = 'state_dicts/test-cifar-10/checkpoint-339'
    model = ViTForImageClassification.from_pretrained(checkpoint,
                                                    id2label=data.id2label,
                                                    label2id=data.label2id)
    
    # Load the tokenizer
    processor = ViTImageProcessor.from_pretrained(checkpoint)

    # Evaluate the fine-tuned model on the test set
    args = TrainingArguments(
        f"./state_dicts/test-cifar-10",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='logs',
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    outputs = trainer.predict(test_dataset)
    print("Test set metrics: ", outputs.metrics)

" Attack functions "

def noise_adv_attack_vit(args, device, epsilon=0.1):
    
    # Load the test dataset
    data = CIFAR10ViT()
    train_dataset, val_dataset, test_dataset = data.get_datasets()

    # Wrap the test dataset with the AdversarialDataset
    adv_test_dataset = AdversarialDataset(test_dataset, noise_adversarial_attack)

    # Load the fine-tuned model
    checkpoint = 'state_dicts/vit/checkpoint-339'
    model = ViTForImageClassification.from_pretrained(checkpoint,
                                                    id2label=data.id2label,
                                                    label2id=data.label2id)
    # Evaluate using Trainer
    processor = ViTImageProcessor.from_pretrained(checkpoint)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            f"./state_dicts/test-cifar-10",
            save_strategy="epoch",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=10,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir='logs',
            remove_unused_columns=False,
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda data: {'pixel_values': torch.stack([f['pixel_values'] for f in data]),
                                    'labels': torch.tensor([f['labels'] for f in data])},
        compute_metrics=lambda p: {'accuracy': (p.predictions.argmax(-1) == p.label_ids).astype(float).mean()},
        tokenizer=processor,
    )

    # Use the adversarial test dataset
    outputs = trainer.predict(adv_test_dataset)
    print("Test set metrics: ", outputs.metrics)

def fgsm_adv_attack_vit(args, device, epsilon=0.01):
    
    # Create an adversarial DataLoader using FGSM
    def create_adversarial_test_loader(dataset, model, epsilon=0.1):
        model.eval()
        adv_samples = []

        for data in dataset:
            image, target = data['pixel_values'], data['label']
            image = image.unsqueeze(0).to(device)
            target = torch.tensor([target]).to(device)

            adv_image = fgsm_adversarial_attack(model, image, target, epsilon)
            adv_samples.append((adv_image.squeeze(0).cpu(), target.cpu()))

        adv_dataset = [{'pixel_values': img, 'label': lbl} for img, lbl in adv_samples]
        return DataLoader(adv_dataset, batch_size=32, shuffle=False)


    # Manually evaluate model on adversarial DataLoader
    def evaluate_model(model, dataloader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                images = torch.stack([item['pixel_values'] for item in batch]).to(device)
                labels = torch.tensor([item['labels'] for item in batch]).to(device)

                outputs = model(images)
                _, predicted = outputs.logits.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f'Adversarial Test Accuracy: {accuracy * 100:.2f}%')


    # Load the test dataset and fine-tuned ViT model
    data = CIFAR10ViT()
    train_dataset, val_dataset, test_dataset = data.get_datasets()

    checkpoint = 'state_dicts/vit/checkpoint-339'
    model = ViTForImageClassification.from_pretrained(checkpoint,
                                                    id2label=data.id2label,
                                                    label2id=data.label2id).to(device)

    # Create adversarial test loader and evaluate model
    adv_test_loader = create_adversarial_test_loader(test_dataset, model, epsilon)
    evaluate_model(model, adv_test_loader)

def pgd_adv_attack_vit(args, device, epsilon=0.01):
    
    # Create an adversarial DataLoader using PGD
    def create_adversarial_test_loader(dataset, model, epsilon=0.1):
        model.eval()
        adv_samples = []

        for data in dataset:
            image, target = data['pixel_values'], data['label']
            image = image.unsqueeze(0).to(device)
            target = torch.tensor([target]).to(device)

            adv_image = pgd_adversarial_attack(model, image, target, epsilon=epsilon, alpha=0.01, iters=10)
            adv_samples.append((adv_image.squeeze(0).cpu(), target.cpu()))

        adv_dataset = [{'pixel_values': img, 'label': lbl} for img, lbl in adv_samples]
        return DataLoader(adv_dataset, batch_size=32, shuffle=False)


    # Manually evaluate model on adversarial DataLoader
    def evaluate_model(model, dataloader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                images = torch.stack([item['pixel_values'] for item in batch]).to(device)
                labels = torch.tensor([item['labels'] for item in batch]).to(device)

                outputs = model(images)
                _, predicted = outputs.logits.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f'Adversarial Test Accuracy: {accuracy * 100:.2f}%')


    # Load the test dataset and fine-tuned ViT model
    data = CIFAR10ViT()
    train_dataset, val_dataset, test_dataset = data.get_datasets()

    checkpoint = 'state_dicts/vit/checkpoint-339'
    model = ViTForImageClassification.from_pretrained(checkpoint,
                                                    id2label=data.id2label,
                                                    label2id=data.label2id).to(device)

    # Create adversarial test loader and evaluate model
    adv_test_loader = create_adversarial_test_loader(test_dataset, model, epsilon)
    evaluate_model(model, adv_test_loader)

def cw_adv_attack_vit(args, device, c=0.01):
    
    # Create an adversarial DataLoader using CW
    def create_adversarial_test_loader(dataset, model, c=0.01):
        model.eval()
        adv_samples = []

        for data in dataset:
            image, target = data['pixel_values'], data['label']
            image = image.unsqueeze(0).to(device)
            target = torch.tensor([target]).to(device)

            adv_image = cw_adversarial_attack(model, image, target, c=c, kappa=0, max_iter=500, device=device)
            adv_samples.append((adv_image.squeeze(0).cpu(), target.cpu()))

        adv_dataset = [{'pixel_values': img, 'label': lbl} for img, lbl in adv_samples]
        return DataLoader(adv_dataset, batch_size=32, shuffle=False)
    

    # Manually evaluate model on adversarial DataLoader
    def evaluate_model(model, dataloader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                images = torch.stack([item['pixel_values'] for item in batch]).to(device)
                labels = torch.tensor([item['labels'] for item in batch]).to(device)

                outputs = model(images)
                _, predicted = outputs.logits.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f'Adversarial Test Accuracy: {accuracy * 100:.2f}%')


    # Load the test dataset and fine-tuned ViT model
    data = CIFAR10ViT()
    train_dataset, val_dataset, test_dataset = data.get_datasets()

    checkpoint = 'state_dicts/vit/checkpoint-339'
    model = ViTForImageClassification.from_pretrained(checkpoint,
                                                    id2label=data.id2label,
                                                    label2id=data.label2id).to(device)
    
    # Create adversarial test loader and evaluate model
    adv_test_loader = create_adversarial_test_loader(test_dataset, model, c)
    evaluate_model(model, adv_test_loader)

" Defend functions "

def noise_adv_train_vit(args, device, epsilon=0.1):
    
    # Load the dataset
    data = CIFAR10ViT()
    train_dataset, val_dataset, test_dataset = data.get_datasets()

    # Wrap the train dataset with the AdversarialDataset
    adv_train_dataset = AdversarialDataset(train_dataset, noise_adversarial_attack, epsilon=epsilon)

    # Load the pre-trained ViT model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  id2label=data.id2label,
                                                  label2id=data.label2id).to(device)

    # Fine-tune the pre-trained model
    args = TrainingArguments(
        f"./state_dicts/test-cifar-10",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='logs',
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=adv_train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=data.processor,
    )
    trainer.train()

def fgsm_adv_train_vit(args, device, epsilon=0.01):
        
    # Load the dataset
    data = CIFAR10ViT()
    train_dataset, val_dataset, test_dataset = data.get_datasets()

    # Wrap the train dataset with the AdversarialDataset
    adv_train_dataset = AdversarialDataset(train_dataset, fgsm_adversarial_attack, epsilon=epsilon)

    # Load the pre-trained ViT model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                id2label=data.id2label,
                                                label2id=data.label2id).to(device)

    # Fine-tune the pre-trained model
    args = TrainingArguments(
        f"./state_dicts/test-cifar-10",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='logs',
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=adv_train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=data.processor,
    )
    trainer.train()

def pgd_adv_train_vit(args, device):
        
    # Load the dataset
    data = CIFAR10ViT()
    train_dataset, val_dataset, test_dataset = data.get_datasets()

    # Wrap the train dataset with the AdversarialDataset
    adv_train_dataset = AdversarialDataset(train_dataset, pgd_adversarial_attack, epsilon=0.01, alpha=0.01, iters=10)

    # Load the pre-trained ViT model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                id2label=data.id2label,
                                                label2id=data.label2id).to(device)

    # Fine-tune the pre-trained model
    args = TrainingArguments(
        f"./state_dicts/test-cifar-10",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='logs',
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=adv_train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=data.processor,
    )
    trainer.train()

def cw_adv_train_vit(args, device):

    # Load the dataset
    data = CIFAR10ViT()
    train_dataset, val_dataset, test_dataset = data.get_datasets()

    # Wrap the train dataset with the AdversarialDataset
    adv_train_dataset = AdversarialDataset(train_dataset, cw_adversarial_attack, c=0.01, kappa=0, max_iter=500, device=device)

    # Load the pre-trained ViT model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                id2label=data.id2label,
                                                label2id=data.label2id).to(device)

    # Fine-tune the pre-trained model
    args = TrainingArguments(
        f"./state_dicts/test-cifar-10",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='logs',
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=adv_train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=data.processor,
    )
    trainer.train()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--model", type=str, default=None, help="VGG11 or ViT")
    parser.add_argument("--action", type=str, default=None, help="train or eval")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--log_interval', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}')
    
    if args.model == "VGG11":
    
        if args.action == "train":
            train_vgg(args, device)

        elif args.action == "eval":
            eval_vgg(args, device)

        elif args.action == "noise_adv_attack":
            accs = []
            for i in np.arange(0.0, 1.1, 0.1):
                acc = noise_adv_attack(args, device, epsilon=i)
                accs.append(acc)
            print("acc: ", accs)
        
        elif args.action == "fgsm_adv_attack":
            accs = []
            for i in np.arange(0.0, 1.1, 0.1):
                acc = fgsm_adv_attack(args, device, epsilon=i)
                accs.append(acc)
            print("accs: ", accs)        
        
        elif args.action == "pgd_adv_attack":
            accs = []
            for i in np.arange(0.0, 1.1, 0.1):
                acc = pgd_adv_attack(args, device, epsilon=i)
                accs.append(acc * 100)
            print("accs: ", accs)    
        
        elif args.action == "cw_adv_attack":
            accs = []
            for i in np.arange(0.0, 1.1, 0.1):
                acc = cw_adv_attack(args, device, c=i)
                accs.append(acc * 100)
            print("accs: ", accs)    

        elif args.action == "noise_adv_train":
            accs = []
            total_losses = []
            for i in np.arange(0.0, 1.1, 0.1):
                print("epsilon: ", i)
                acc, train_loss = noise_adv_train(args, device, epsilon=i)
                accs.append(acc * 100)
                total_losses.append(train_loss)
            print("accs: ", accs)
            print("train_loss: ", total_losses)

        elif args.action == "fgsm_adv_train":
            accs = []
            total_losses = []
            for i in np.arange(0.0, 1.1, 0.1):
                acc, train_loss = fgsm_adv_train(args, device, epsilon=i)
                accs.append(acc * 100)
                total_losses.append(train_loss)
            print("accs: ", accs)
            print("train_loss: ", total_losses)
        
        elif args.action == "pgd_adv_train":
            pgd_adv_train(args, device)

        elif args.action == "cw_adv_train":
            cw_adv_train(args, device)

        else:
            print("Invalid action.")

    elif args.model == "ViT":

        if args.action == "train":
            train_vit()

        elif args.action == "eval":
            eval_vit()

        elif args.action == "noise_adv_attack_vit":
            noise_adv_attack_vit(args, device, epsilon=0.1)

        elif args.action == "fgsm_adv_attack_vit":
            fgsm_adv_attack_vit(args, device, epsilon=0.01)

        elif args.action == "pgd_adv_attack_vit":
            pgd_adv_attack_vit(args, device, epsilon=0.01)

        elif args.action == "cw_adv_attack_vit":
            cw_adv_attack_vit(args, device, c=0.01)

        elif args.action == "noise_adv_train_vit":
            noise_adv_train_vit(args, device, epsilon=0.1)
        
        elif args.action == "fgsm_adv_train_vit":
            fgsm_adv_train_vit(args, device, epsilon=0.01)

        elif args.action == "pgd_adv_train_vit":
            pgd_adv_train_vit(args, device)

        elif args.action == "cw_adv_train_vit":
            cw_adv_train_vit(args, device)

        else:
            print("Invalid action.")

    else:
        print("Invalid model. Use either 'VGG11' or 'ViT'.")

