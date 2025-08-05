import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Parameters
num_users = 1000
num_products = 500
num_orders = 5000
num_aisles = 20
num_departments = 10

# Generate users
user_ids = np.arange(1, num_users + 1)

# Generate products
product_ids = np.arange(1, num_products + 1)
product_names = [f'Product_{i}' for i in product_ids]
products = pd.DataFrame({
    'product_id': product_ids,
    'product_name': product_names,
    'aisle_id': np.random.randint(1, num_aisles + 1, size=num_products),
    'department_id': np.random.randint(1, num_departments + 1, size=num_products),
})

# Generate aisles
aisles = pd.DataFrame({
    'aisle_id': np.arange(1, num_aisles + 1),
    'aisle': [f'Aisle_{i}' for i in range(1, num_aisles + 1)],
})

# Generate departments
departments = pd.DataFrame({
    'department_id': np.arange(1, num_departments + 1),
    'department': [f'Department_{i}' for i in range(1, num_departments + 1)],
})

# Generate orders
order_ids = np.arange(1, num_orders + 1)
orders = pd.DataFrame({
    'order_id': order_ids,
    'user_id': np.random.choice(user_ids, size=num_orders),
    'order_number': 1,
    'order_dow': np.random.randint(0, 7, size=num_orders),
    'order_hour_of_day': np.random.randint(0, 24, size=num_orders),
    'days_since_prior_order': np.random.randint(1, 30, size=num_orders)
})

# For simplicity, assign order numbers per user
orders['order_number'] = orders.groupby('user_id').cumcount() + 1

# Generate order_products_prior (historical orders)
num_prior = int(num_orders * 0.8)
prior_order_ids = np.random.choice(order_ids, size=num_prior, replace=False)
prior_products = []

for order_id in prior_order_ids:
    n_products = np.random.randint(1, 10)
    products_in_order = np.random.choice(product_ids, size=n_products, replace=False)
    for p in products_in_order:
        prior_products.append({
            'order_id': order_id,
            'product_id': p,
            'add_to_cart_order': np.random.randint(1, 20),
            'reordered': np.random.choice([0, 1], p=[0.7, 0.3])
        })

order_products_prior = pd.DataFrame(prior_products)

# Generate order_products_train (for last orders)
train_order_ids = order_ids[-int(0.2 * num_orders):]
train_products = []

for order_id in train_order_ids:
    n_products = np.random.randint(1, 10)
    products_in_order = np.random.choice(product_ids, size=n_products, replace=False)
    for p in products_in_order:
        train_products.append({
            'order_id': order_id,
            'product_id': p,
            'add_to_cart_order': np.random.randint(1, 20),
            'reordered': np.random.choice([0, 1], p=[0.6, 0.4])
        })

order_products_train = pd.DataFrame(train_products)

# Save datasets to CSV
products.to_csv('products.csv', index=False)
aisles.to_csv('aisles.csv', index=False)
departments.to_csv('departments.csv', index=False)
orders.to_csv('orders.csv', index=False)
order_products_prior.to_csv('order_products_prior.csv', index=False)
order_products_train.to_csv('order_products_train.csv', index=False)

print("Synthetic datasets generated and saved as CSV files.")