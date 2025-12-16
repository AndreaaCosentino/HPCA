import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
"""
df = pd.read_csv(
    "results_part_1.txt", 
    sep=r'\s+',      
    header=None,       
    names=["Maturity","P","F"],
    index_col=False   
)

plt.xticks(
    np.arange(df['Maturity'].min(), df['Maturity'].max() + 0.5, 0.5)  # step = 0.1
)
plt.plot( df['Maturity'],df['F'], marker='o', linestyle='-', color='blue')
plt.xlabel("Maturity")
plt.ylabel("F")
plt.title("Zero coupon bond price, range [0,10]")
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.25)
plt.savefig("F_simulation.pdf")
plt.show()
"""

df = pd.read_csv(
    "theta_values2.txt", 
    sep=r'\s+',      
    header=None,       
    names=["t","theta","theta_2"],
    index_col=False   
)

plt.xticks(
    np.arange(df['t'].min(), df['t'].max() + 1, 1)  # step = 0.1
)
plt.plot( df['t'],df['theta'],  linestyle='-', color='blue')
plt.plot( df['t'],df['theta_2'], linestyle='-', color='red')
plt.xlabel("Time (t)")
plt.ylabel("Theta")
plt.title("Value of theta")
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.25)
plt.savefig("theta.pdf")
plt.show()

