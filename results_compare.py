import pandas as pd

s1 = pd.read_csv("results/scenarios/s1_class1_90/rl_agent_metrics.csv")
s2 = pd.read_csv("results/scenarios/s1_class2_90/rl_agent_metrics.csv")
s3 = pd.read_csv("results/scenarios/s1_class3_90/rl_agent_metrics.csv")

print(f"Class 1 90% - Mean QoE: {s1['QoE'].mean():.6f}")
print(f"Class 2 90% - Mean QoE: {s2['QoE'].mean():.6f}")
print(f"Class 3 90% - Mean QoE: {s3['QoE'].mean():.6f}")
