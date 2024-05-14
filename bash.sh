# Train the VGG11 on CIFAR10 dataset
# python main.py --model VGG11 --action train --batch_size 64 --num_epochs 150 --weight_decay 0

# Evaluate the VGG11 on CIFAR10 dataset
# python main.py --model VGG11 --action eval --batch_size 64

# VGG11
# Apply noise_adv_attack on the VGG11 model
# python main.py --model VGG11 --action noise_adv_attack --batch_size 64

# Apply fgsm_adv_attack on the VGG11 model
# python main.py --model VGG11 --action fgsm_adv_attack --batch_size 64

# Apply pgd_adv_attack on the VGG11 model
# python main.py --model VGG11 --action pgd_adv_attack --batch_size 64

# Apply cw_adv_attack on the VGG11 model
# python main.py --model VGG11 --action cw_adv_attack --batch_size 64

# ViT
# Fine-tune the pre-trained ViT on CIFAR10 dataset
# python main.py --model ViT --action train

# Evaluate the pre-trained ViT on CIFAR10 dataset
# python main.py --model ViT --action eval

# Apply noise_adv_attack on the ViT model
# python main.py --model ViT --action noise_adv_attack_vit --batch_size 64

# Apply fgsm_adv_attack on the ViT model
# python main.py --model ViT --action fgsm_adv_attack_vit --batch_size 64
