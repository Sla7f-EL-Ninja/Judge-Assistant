import json
from collections import defaultdict

with open('all_traces.json', 'r', encoding='utf-8') as f:
    traces = json.load(f)

# Dictionary to store cost per component name
cost_by_name = defaultdict(float)
count_by_name = defaultdict(int)

for trace in traces:
    name = trace.get('name', 'Unknown')
    # Get cost, handle strings or missing values
    cost_str = trace.get('total_cost')
    cost = float(cost_str) if cost_str is not None else 0.0
    
    cost_by_name[name] += cost
    count_by_name[name] += 1

# Sort by cost (highest first)
sorted_costs = sorted(cost_by_name.items(), key=lambda x: x[1], reverse=True)

print(f"{'Component Name':<30} | {'Runs':<10} | {'Cost':<10}")
print("-" * 55)

for name, total in sorted_costs:
    if total > 0.0001: # Only show components that actually cost something
        print(f"{name:<30} | {count_by_name[name]:<10} | ${total:.4f}")

total_sum = sum(cost_by_name.values())
print("-" * 55)
print(f"Total Analyzed: ${total_sum:.4f}")