import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# Load dataset
df = pd.read_csv("placement_predict_50k Dataset (3)(in).csv")



print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# Split dataset into training and testing data
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["PlacementStatus"]
)


tier_order = ["Tier3", "Tier2", "Tier1"]
cgpa_tier_order = ["Low", "Mid", "High"]


ord_enc = OrdinalEncoder(
    categories=[
        tier_order,
        cgpa_tier_order
    ]
)


ordinal_cols = [  "CollegeTier","CGPA_Tier"]


train_df[
    ["CollegeTier_enc", "CGPA_Tier_enc"]] = ord_enc.fit_transform( train_df[ordinal_cols])


# Transform testing data using the same encoder
test_df[
    ["CollegeTier_enc", "CGPA_Tier_enc"]
] = ord_enc.transform(
    test_df[ordinal_cols]
)


# Display results
print("\nTraining data:")
print(train_df.head())

print("\nTesting data:")
print(test_df.head())

print("\nEncoded training columns:")
print(train_df[
    ["CollegeTier", "CollegeTier_enc",
     "CGPA_Tier", "CGPA_Tier_enc"]
].head())

print("\nEncoded testing columns:")
print(test_df[
    ["CollegeTier", "CollegeTier_enc",
     "CGPA_Tier", "CGPA_Tier_enc"]
].head())