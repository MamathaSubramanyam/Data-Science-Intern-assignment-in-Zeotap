# Feature engineering: Aggregate transaction data for each customer
customer_features = transactions_full.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'Price': 'mean',
    'Category': lambda x: x.mode()[0]  # Mode of purchased categories
}).reset_index()

# Encode categorical features (Category)
encoder = OneHotEncoder()
category_encoded = encoder.fit_transform(customer_features[['Category']]).toarray()
category_encoded_df = pd.DataFrame(category_encoded, columns=encoder.get_feature_names_out(['Category']))

# Merge encoded features
customer_features = pd.concat([customer_features, category_encoded_df], axis=1)
customer_features.drop(columns=['Category'], inplace=True)

# Normalize numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_features.iloc[:, 1:])

# Compute cosine similarity
similarity_matrix = cosine_similarity(scaled_features)

# Generate top 3 lookalike recommendations
lookalike_results = {}
for i, customer_id in enumerate(customer_features['CustomerID']):
    similar_indices = np.argsort(similarity_matrix[i])[::-1][1:4]  # Top 3 similar customers
    similar_customers = customer_features['CustomerID'].iloc[similar_indices]
    scores = similarity_matrix[i, similar_indices]
    lookalike_results[customer_id] = list(zip(similar_customers, scores))

# Save to Lookalike.csv
lookalike_df = pd.DataFrame([
    {'CustomerID': cust_id, 'Lookalikes': lookalikes}
    for cust_id, lookalikes in lookalike_results.items()
])
lookalike_df.to_csv('Lookalike.csv', index=False)