import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Sample Dataset (Transactions)
data = {'Milk': [1, 0, 1, 1, 0],
        'Bread': [1, 1, 1, 0, 1],
        'Butter': [0, 1, 1, 1, 1],
        'Eggs': [1, 1, 0, 1, 0]}

df = pd.DataFrame(data)

# Step 2: Apply Apriori Algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# Step 3: Generate Association Rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Step 4: Display Results
print("Frequent Itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", rules)
