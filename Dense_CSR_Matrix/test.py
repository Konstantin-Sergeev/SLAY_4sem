import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('performance_results.csv')

plt.figure(figsize=(10, 6))
for name, group in df.groupby('Matrix Type'):
    if name == 'Dense':
        plt.plot(group['Size'], group['Time'], 'o-', label=name)
    else:
        for sparsity, sub_group in group.groupby('Sparsity'):
            plt.plot(sub_group['Size'], sub_group['Time'], 'o-', 
                    label=f"{name} (sparsity={sparsity})")

plt.xlabel('Matrix Size')
plt.ylabel('Time (s)')
plt.title('Matrix-Vector Multiplication Performance')
plt.legend()
plt.grid(True)
plt.savefig('performance_comparison.png')
plt.show()
