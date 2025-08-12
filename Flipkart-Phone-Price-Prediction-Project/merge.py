import pandas as pd

df1 = pd.read_csv("5G_phone.csv")
df2 = pd.read_csv("phones_data.csv")

# Combine both datasets
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save as new file
combined_df.to_csv("phone.csv", index=False)
