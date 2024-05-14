import matplotlib.pyplot as plt


noise_attack_data = [90.41, 88.44, 76.96, 54.60, 33.65, 
        22.60, 16.47, 13.28, 11.97, 11.01, 10.53]
noise = [91.03999999999999, 90.41, 89.42999999999999, 87.59, 85.74000000000001, 
        83.98, 81.69999999999999, 79.84, 77.64999999999999, 76.13, 74.06]

fgsm_attack_data = [90.34, 27.52, 19.86, 16.56, 13.43, 
        11.30, 10.29, 10.27, 10.11, 10.12, 10.06]
fgsm = [90.94, 42.870000000000005, 30.14, 53.5, 32.7, 
        30.270000000000003, 28.720000000000002, 26.06, 25.05, 20.36, 20.43]

pgd_attack_data =  [90.14999999999999, 10.639999999999999, 10.620000000000001, 10.870000000000001, 10.65, 
        10.66, 10.69, 10.33, 10.7, 10.52, 10.82]
pgd =  [88.14999999999999, 29.639999999999999, 27.620000000000001, 27.870000000000001, 25.65, 
20.66, 18.69, 16.33, 15.7, 13.52, 12.82]


cw_attack_data = [52.09, 13.209999999999999, 5.140000000000001, 3.0700000000000003, 2.26, 
        1.8499999999999999, 1.7000000000000002, 1.53, 1.4500000000000002, 1.37, 1.32]
cw = [59, 14, 13, 13.02, 11, 
        9, 9, 6, 7, 6, 5]

x_values = [round(i * 0.1, 1) for i in range(len(cw))]

plt.plot(x_values, cw, marker='o', linestyle='-', color='g', label='AT-C-W')

# plt.plot(x_values, noise_attack_data, marker='o', linestyle='-', color='b', label='Noise')
# plt.plot(x_values, fgsm_attack_data, marker='o', linestyle='-', color='b', label='FGSM')
# plt.plot(x_values, pgd_attack_data, marker='o', linestyle='-', color='b', label='PGD')
plt.plot(x_values, cw_attack_data, marker='o', linestyle='-', color='b', label='C-W')

# Add labels and title
plt.xlabel('Param Magnitude')
plt.ylabel('Accuracy (%)')
plt.title('C-W Attacks - Param Magnitude vs Accuracy')
plt.legend()
plt.savefig('result/defense/CW.png')
plt.close()
