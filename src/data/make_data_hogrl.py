import pandas as pd
import numpy as np
import os
import networkx as nx
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt

def load_raw_data(file_path="D:\\fraud_detection\\data\\raw\\data.csv"):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    df = df[df['type'].isin(['CASH_OUT', 'TRANSFER', 'PAYMENT'])]
    
    # Remove transactions with zero amount
    df = df[df['amount'] > 0].reset_index(drop=True)

    # Keep 1% of the data to save time and keep all rows with isFraud = 1
    df_fraud = df[df['isFraud'] == 1]
    df_non_fraud = df[df['isFraud'] == 0].sample(frac=0.01, random_state=42)
    df = pd.concat([df_fraud, df_non_fraud], ignore_index=True)
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate balance differences
    df['balance_diff_Org'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_diff_Dest'] = df['oldbalanceDest'] - df['newbalanceDest']
    
    # Clean and convert client IDs
    df['nameOrig_clean'] = df['nameOrig'].str.replace('C', '')
    df['nameDest_clean'] = df['nameDest'].str.replace('C', '')
    
    # Flag merchant accounts
    df['is_orig_merchant'] = df['nameOrig'].str.startswith('M').astype(int)
    df['is_dest_merchant'] = df['nameDest'].str.startswith('M').astype(int)
      # Map transaction types to integers
    df['type_encoded'] = df['type'].map({'CASH_OUT': 0, 'TRANSFER': 1, 'PAYMENT': 2})
    
    print(f"Data loaded and preprocessed. Total transactions: {len(df)}")
    return df

def create_graph_data(df):
    print("Creating graph data...")
    
    # Create node features
    client_nodes = {}
    merchant_nodes = {}
    
    # Process originating accounts
    for idx, row in df.iterrows():
        if row['nameOrig'].startswith('C'):  # Client account
            client_id = row['nameOrig']
            if client_id not in client_nodes:
                client_nodes[client_id] = {
                    'total_sent': 0,
                    'total_transactions': 0,
                    'avg_amount': 0,
                    'max_amount': 0,
                    'min_balance': float('inf'),
                    'fraud_involved': 0
                }
            
            client_nodes[client_id]['total_sent'] += row['amount']
            client_nodes[client_id]['total_transactions'] += 1
            client_nodes[client_id]['max_amount'] = max(client_nodes[client_id]['max_amount'], row['amount'])
            client_nodes[client_id]['min_balance'] = min(client_nodes[client_id]['min_balance'], row['newbalanceOrig'])
            client_nodes[client_id]['fraud_involved'] = max(client_nodes[client_id]['fraud_involved'], row['isFraud'])

        elif row['nameOrig'].startswith('M'):  # Merchant account
            merchant_id = row['nameOrig']
            if merchant_id not in merchant_nodes:
                merchant_nodes[merchant_id] = {
                    'total_sent': 0,
                    'total_transactions': 0
                }
            
            merchant_nodes[merchant_id]['total_sent'] += row['amount']
            merchant_nodes[merchant_id]['total_transactions'] += 1
    
    # Process destination accounts
    for idx, row in df.iterrows():
        if row['nameDest'].startswith('C'):  # Client account
            client_id = row['nameDest']
            if client_id not in client_nodes:
                client_nodes[client_id] = {
                    'total_received': 0,
                    'received_transactions': 0,
                    'max_received': 0,
                    'fraud_involved': 0,
                    'total_sent': 0,
                    'total_transactions': 0,
                    'max_amount': 0,
                    'min_balance': float('inf')
                }
            
            # Add received fields if they don't exist
            if 'total_received' not in client_nodes[client_id]:
                client_nodes[client_id]['total_received'] = 0
                client_nodes[client_id]['received_transactions'] = 0
                client_nodes[client_id]['max_received'] = 0
            
            client_nodes[client_id]['total_received'] += row['amount']
            client_nodes[client_id]['received_transactions'] += 1
            client_nodes[client_id]['max_received'] = max(client_nodes[client_id].get('max_received', 0), row['amount'])
            client_nodes[client_id]['fraud_involved'] = max(client_nodes[client_id].get('fraud_involved', 0), row['isFraud'])
            
        elif row['nameDest'].startswith('M'):  # Merchant account
            merchant_id = row['nameDest']
            if merchant_id not in merchant_nodes:
                merchant_nodes[merchant_id] = {
                    'total_received': 0,
                    'received_transactions': 0
                }
            
            # Add received fields if they don't exist
            if 'total_received' not in merchant_nodes[merchant_id]:
                merchant_nodes[merchant_id]['total_received'] = 0
                merchant_nodes[merchant_id]['received_transactions'] = 0
            
            merchant_nodes[merchant_id]['total_received'] += row['amount']
            merchant_nodes[merchant_id]['received_transactions'] += 1
    
    # Finalize node features
    for client_id in client_nodes:
        if 'total_transactions' in client_nodes[client_id] and client_nodes[client_id]['total_transactions'] > 0:
            client_nodes[client_id]['avg_amount'] = client_nodes[client_id]['total_sent'] / client_nodes[client_id]['total_transactions']
        
        # Ensure all clients have the same feature set
        client_nodes[client_id]['total_sent'] = client_nodes[client_id].get('total_sent', 0)
        client_nodes[client_id]['total_transactions'] = client_nodes[client_id].get('total_transactions', 0)
        client_nodes[client_id]['avg_amount'] = client_nodes[client_id].get('avg_amount', 0)
        client_nodes[client_id]['max_amount'] = client_nodes[client_id].get('max_amount', 0)
        client_nodes[client_id]['min_balance'] = client_nodes[client_id].get('min_balance', float('inf'))
        if client_nodes[client_id]['min_balance'] == float('inf'):
            client_nodes[client_id]['min_balance'] = 0
        client_nodes[client_id]['total_received'] = client_nodes[client_id].get('total_received', 0)
        client_nodes[client_id]['received_transactions'] = client_nodes[client_id].get('received_transactions', 0)
        client_nodes[client_id]['max_received'] = client_nodes[client_id].get('max_received', 0)
        client_nodes[client_id]['fraud_involved'] = client_nodes[client_id].get('fraud_involved', 0)
    
    for merchant_id in merchant_nodes:
        # Ensure all merchants have the same feature set
        merchant_nodes[merchant_id]['total_sent'] = merchant_nodes[merchant_id].get('total_sent', 0)
        merchant_nodes[merchant_id]['total_transactions'] = merchant_nodes[merchant_id].get('total_transactions', 0)
        merchant_nodes[merchant_id]['total_received'] = merchant_nodes[merchant_id].get('total_received', 0)
        merchant_nodes[merchant_id]['received_transactions'] = merchant_nodes[merchant_id].get('received_transactions', 0)
    
    # Create edges (transactions)
    edges = []
    for idx, row in df.iterrows():
        edges.append({
            'source': row['nameOrig'],
            'target': row['nameDest'],
            'amount': row['amount'],
            'type': row['type_encoded'],
            'oldbalanceOrg': row['oldbalanceOrg'],
            'newbalanceOrig': row['newbalanceOrig'],
            'oldbalanceDest': row['oldbalanceDest'],
            'newbalanceDest': row['newbalanceDest'],
            'isFraud': row['isFraud'],
            'balance_diff_Org': row['balance_diff_Org'],
            'balance_diff_Dest': row['balance_diff_Dest'],
            'step': row['step']
        })
    
    print(f"Graph data created. Client nodes: {len(client_nodes)}, Merchant nodes: {len(merchant_nodes)}, Edges: {len(edges)}")
    
    return {
        'client_nodes': client_nodes,
        'merchant_nodes': merchant_nodes,
        'edges': edges
    }

def normalize_features(graph_data):
    """
    Normalize node and edge features for better model performance
    """
    print("Normalizing features...")
    
    # Normalize client node features
    client_features = pd.DataFrame.from_dict(graph_data['client_nodes'], orient='index')
    scaler_client = StandardScaler()
    client_features_scaled = scaler_client.fit_transform(client_features)
    client_features_scaled_df = pd.DataFrame(client_features_scaled, index=client_features.index, columns=client_features.columns)
    
    graph_data['client_nodes_normalized'] = client_features_scaled_df.to_dict(orient='index')
    
    # Normalize merchant node features if they exist
    if len(graph_data['merchant_nodes']) > 0:
        merchant_features = pd.DataFrame.from_dict(graph_data['merchant_nodes'], orient='index')
        scaler_merchant = StandardScaler()
        merchant_features_scaled = scaler_merchant.fit_transform(merchant_features)
        merchant_features_scaled_df = pd.DataFrame(merchant_features_scaled, index=merchant_features.index, columns=merchant_features.columns)
        graph_data['merchant_nodes_normalized'] = merchant_features_scaled_df.to_dict(orient='index')
        scaler_merchant = scaler_merchant
    else:
        print("No merchant nodes found, skipping merchant feature normalization")
        graph_data['merchant_nodes_normalized'] = {}
        scaler_merchant = None    # Normalize edge features (exclude categorical features)
    edge_features = pd.DataFrame(graph_data['edges'])
    categorical_cols = ['source', 'target', 'type', 'isFraud', 'step']
    numerical_cols = [col for col in edge_features.columns if col not in categorical_cols]
    
    scaler_edge = StandardScaler()
    edge_features[numerical_cols] = scaler_edge.fit_transform(edge_features[numerical_cols])
    
    graph_data['edges_normalized'] = edge_features.to_dict(orient='records')
    graph_data['scalers'] = {
        'client': scaler_client,
        'merchant': scaler_merchant,
        'edge': scaler_edge
    }
    
    print("Features normalized")
    return graph_data

def create_networkx_graph(graph_data):
    """
    Create a NetworkX heterogeneous graph for visualization and analysis
    """
    print("Creating NetworkX graph...")
    
    G = nx.DiGraph()
    
    # Add client nodes
    for client_id, features in graph_data['client_nodes'].items():
        G.add_node(client_id, **features, node_type='client')
    
    # Add merchant nodes
    for merchant_id, features in graph_data['merchant_nodes'].items():
        G.add_node(merchant_id, **features, node_type='merchant')
    
    # Add edges
    for edge in graph_data['edges']:
        G.add_edge(edge['source'], edge['target'], **{k: v for k, v in edge.items() if k not in ['source', 'target']})
    
    print(f"NetworkX graph created. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G

def prepare_train_val_test_split(graph_data, val_ratio=0.1, test_ratio=0.2):
    """
    Prepare train, validation, and test splits for HOGRL
    """
    print("Preparing train/val/test split...")
    
    # Get fraudulent and non-fraudulent edges
    edges_df = pd.DataFrame(graph_data['edges'])
    fraud_edges = edges_df[edges_df['isFraud'] == 1]
    non_fraud_edges = edges_df[edges_df['isFraud'] == 0]
    
    # Calculate split sizes
    n_fraud = len(fraud_edges)
    n_fraud_test = int(n_fraud * test_ratio)
    n_fraud_val = int(n_fraud * val_ratio)
    n_fraud_train = n_fraud - n_fraud_test - n_fraud_val
    
    n_non_fraud = len(non_fraud_edges)
    n_non_fraud_test = int(n_non_fraud * test_ratio)
    n_non_fraud_val = int(n_non_fraud * val_ratio)
    n_non_fraud_train = n_non_fraud - n_non_fraud_test - n_non_fraud_val
    
    # Shuffle and split
    fraud_edges = fraud_edges.sample(frac=1, random_state=42).reset_index(drop=True)
    non_fraud_edges = non_fraud_edges.sample(frac=1, random_state=42).reset_index(drop=True)
    
    fraud_train = fraud_edges.iloc[:n_fraud_train]
    fraud_val = fraud_edges.iloc[n_fraud_train:n_fraud_train+n_fraud_val]
    fraud_test = fraud_edges.iloc[n_fraud_train+n_fraud_val:]
    
    non_fraud_train = non_fraud_edges.iloc[:n_non_fraud_train]
    non_fraud_val = non_fraud_edges.iloc[n_non_fraud_train:n_non_fraud_train+n_non_fraud_val]
    non_fraud_test = non_fraud_edges.iloc[n_non_fraud_train+n_non_fraud_val:]
    
    # Combine fraud and non-fraud
    train_edges = pd.concat([fraud_train, non_fraud_train]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_edges = pd.concat([fraud_val, non_fraud_val]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_edges = pd.concat([fraud_test, non_fraud_test]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Split created. Train: {len(train_edges)}, Val: {len(val_edges)}, Test: {len(test_edges)}")
    print(f"Fraud distribution - Train: {len(fraud_train)}/{len(train_edges)}, Val: {len(fraud_val)}/{len(val_edges)}, Test: {len(fraud_test)}/{len(test_edges)}")
    
    return {
        'train': train_edges.to_dict(orient='records'),
        'val': val_edges.to_dict(orient='records'),
        'test': test_edges.to_dict(orient='records')
    }

def save_hogrl_data(graph_data, data_splits, output_dir="D:\\fraud_detection\\data\\data_process\\hogrl"):
    """
    Save preprocessed data for HOGRL
    """
    print(f"Saving data to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)      # Save node features
    client_features = pd.DataFrame.from_dict(graph_data['client_nodes_normalized'], orient='index')
    client_features.to_csv(os.path.join(output_dir, 'client_features.csv'), index_label='node_id')
    
    # Only save merchant features if they exist
    if len(graph_data['merchant_nodes_normalized']) > 0:
        merchant_features = pd.DataFrame.from_dict(graph_data['merchant_nodes_normalized'], orient='index')
        merchant_features.to_csv(os.path.join(output_dir, 'merchant_features.csv'), index_label='node_id')
        print(f"Merchant features saved with {len(merchant_features)} merchants")
    else:
        print("No merchant features to save (empty dictionary)")
        # Create an empty file with headers to maintain consistency
        pd.DataFrame(columns=['total_sent', 'total_transactions', 'total_received', 'received_transactions']).to_csv(os.path.join(output_dir, 'merchant_features.csv'), index_label='node_id')
      # Save edge splits
    train_edges = pd.DataFrame(data_splits['train'])
    train_edges.to_csv(os.path.join(output_dir, 'train_edges.csv'), index=False)
    
    val_edges = pd.DataFrame(data_splits['val'])
    val_edges.to_csv(os.path.join(output_dir, 'val_edges.csv'), index=False)
    
    test_edges = pd.DataFrame(data_splits['test'])
    test_edges.to_csv(os.path.join(output_dir, 'test_edges.csv'), index=False)
    
    # Save the graph structure for later use
    nx_graph = create_networkx_graph(graph_data)
    with open(os.path.join(output_dir, 'graph.pkl'), 'wb') as f:
        pickle.dump(nx_graph, f)
      # Save metadata
    metadata = {
        'num_client_nodes': len(graph_data['client_nodes']),
        'num_merchant_nodes': len(graph_data['merchant_nodes']),
        'num_edges': len(graph_data['edges']),
        'client_features': list(client_features.columns),
        'merchant_features': list(merchant_features.columns) if 'merchant_features' in locals() and len(merchant_features) > 0 else [],
        'edge_features': list(train_edges.columns),
        'fraud_ratio': train_edges['isFraud'].mean()
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save scalers for future use
    with open(os.path.join(output_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(graph_data['scalers'], f)
    
    print(f"Data saved successfully to {output_dir}")

def plot_graph_statistics(graph, output_dir="D:\\fraud_detection\\src\\images"):
    """
    Plot some basic statistics of the graph
    """
    print("Plotting graph statistics...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Node type distribution
    node_types = [graph.nodes[node]['node_type'] for node in graph.nodes() if 'node_type' in graph.nodes[node]]
    node_type_counts = pd.Series(node_types).value_counts()
    
    plt.figure(figsize=(10, 6))
    node_type_counts.plot(kind='bar')
    plt.title('Distribution of Node Types')
    plt.xlabel('Node Type')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'hogrl_node_types.png'))
    
    # Fraud distribution in edges
    fraud_status = [graph.edges[edge]['isFraud'] for edge in graph.edges()]
    fraud_counts = pd.Series(fraud_status).value_counts()
    
    plt.figure(figsize=(10, 6))
    fraud_counts.plot(kind='bar')
    plt.title('Distribution of Fraud in Transactions')
    plt.xlabel('Is Fraud')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'hogrl_fraud_distribution.png'))
    
    # Degree distribution
    in_degrees = [graph.in_degree(node) for node in graph.nodes()]
    out_degrees = [graph.out_degree(node) for node in graph.nodes()]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(in_degrees, bins=50)
    plt.title('In-Degree Distribution')
    plt.xlabel('In-Degree')
    plt.ylabel('Count')
    plt.xscale('log')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.hist(out_degrees, bins=50)
    plt.title('Out-Degree Distribution')
    plt.xlabel('Out-Degree')
    plt.ylabel('Count')
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hogrl_degree_distribution.png'))
    
    print(f"Statistics plots saved to {output_dir}")

def main():
    """
    Main function to run the HOGRL data preparation pipeline
    """
    # Load and preprocess raw data
    df = load_raw_data()
    
    # Create graph data
    graph_data = create_graph_data(df)
    
    # Normalize features
    graph_data = normalize_features(graph_data)
    
    # Create NetworkX graph
    nx_graph = create_networkx_graph(graph_data)
    
    # Plot graph statistics
    plot_graph_statistics(nx_graph)
    
    # Prepare train/val/test splits
    data_splits = prepare_train_val_test_split(graph_data)
    
    # Save data for HOGRL
    save_hogrl_data(graph_data, data_splits)
    
    print("HOGRL data preparation completed successfully!")

if __name__ == "__main__":
    main()