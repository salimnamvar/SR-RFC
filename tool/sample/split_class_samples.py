import pandas as pd
from sklearn.model_selection import train_test_split

train = 'G:/Challenges/RNA/data/train_data.parquet'

df = pd.read_parquet(train)

# Extract the first and third columns (0-based index)
df.iloc[:, 0] = df.index
selected_columns = df.iloc[:, [0, 2]]

# Split the DataFrame into two separate DataFrames based on the 'experiment_type' column
df_2A3_MaP = selected_columns[selected_columns['experiment_type'] == '2A3_MaP']
df_DMS_MaP = selected_columns[selected_columns['experiment_type'] == 'DMS_MaP']

# Shuffle the data
df_2A3_MaP = df_2A3_MaP.sample(frac=1, random_state=42)
df_DMS_MaP = df_DMS_MaP.sample(frac=1, random_state=42)

# Split each DataFrame into train and validation sets
train_2A3_MaP, val_2A3_MaP = train_test_split(df_2A3_MaP, test_size=0.3, random_state=42)
train_DMS_MaP, val_DMS_MaP = train_test_split(df_DMS_MaP, test_size=0.3, random_state=42)

# Save the training and validation sets as separate CSV files without headers
train_2A3_MaP[['sequence_id']].to_csv('train_2A3_MaP.csv', index=False, header=False)
val_2A3_MaP[['sequence_id']].to_csv('val_2A3_MaP.csv', index=False, header=False)
train_DMS_MaP[['sequence_id']].to_csv('train_DMS_MaP.csv', index=False, header=False)
val_DMS_MaP[['sequence_id']].to_csv('val_DMS_MaP.csv', index=False, header=False)
