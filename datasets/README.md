# Datasets

This directory contains datasets for training and evaluating CICRec.

## Directory Structure

```
datasets/
└── sequential/
    └── mooc/
        ├── train.tsv                        # Training sequences
        ├── test.tsv                         # Test sequences
        └── course_easyrec_embeddings.pt     # Pre-trained embeddings (optional)
```

## Dataset Format

### Sequential Recommendation Data

**File Format**: Tab-separated values (TSV)

**train.tsv / test.tsv**:
```
session_id    item_id_seq    item_id
0             1 2 3 4         5
1             2 3 4           6
2             5 6 7 8 9       10
```

**Columns**:
1. `session_id`: User or session identifier (integer)
2. `item_id_seq`: Space-separated sequence of item IDs (integers)
3. `item_id`: Next item to predict (integer, ground truth)

**Requirements**:
- Item IDs should start from 1 (0 is reserved for padding)
- Sequences can be of variable length
- No header row in the actual data (first line is header)

### Semantic Embeddings (Optional)

**File**: `course_easyrec_embeddings.pt`

**Format**: PyTorch tensor saved with `torch.save()`

**Shape**: `[num_items + 1, 1024]`
- Index 0: Padding (should be zeros)
- Index 1 to num_items: Item embeddings

**How to create**:
```python
import torch

# Assuming you have embeddings for items 1 to N
item_embeddings = ...  # Shape: [N, 1024]

# Add padding at index 0
embeddings_with_padding = torch.cat([
    torch.zeros(1, 1024),  # Padding
    item_embeddings
], dim=0)

# Save
torch.save(embeddings_with_padding, 'course_easyrec_embeddings.pt')
```

## MOOC Dataset

The MOOC (Massive Open Online Course) dataset contains user learning sequences.

### Statistics
- Users: [To be filled]
- Items (Courses): [To be filled]
- Interactions: [To be filled]
- Avg. sequence length: [To be filled]
- Sparsity: [To be filled]

### Data Collection
[Describe how the data was collected]

### Preprocessing
[Describe preprocessing steps]

### Citation
If you use this dataset, please cite:
```bibtex
[Dataset citation]
```

## Adding Your Own Dataset

1. Create a new directory: `datasets/sequential/your_dataset/`

2. Prepare `train.tsv` and `test.tsv` following the format above

3. (Optional) Prepare semantic embeddings: `your_embeddings.pt`

4. Update the config:
   ```bash
   python main.py --model cicrec --dataset your_dataset
   ```

5. Adjust hyperparameters in `config/modelconf/cicrec.yml` if needed

## Data Splitting

### Recommended Split
- Training: 80%
- Test: 20%

### Temporal Split (Recommended for sequential data)
```python
# For each user
sequences = user_sequences
train_seq = sequences[:-1]  # All but last
test_seq = sequences[-1]     # Last item
```

### Example Preprocessing Script

```python
import pandas as pd

def create_sequential_data(interactions_df):
    """
    Convert interaction data to sequential format
    
    Args:
        interactions_df: DataFrame with columns [user_id, item_id, timestamp]
    
    Returns:
        train_data, test_data: DataFrames in TSV format
    """
    # Sort by user and timestamp
    interactions_df = interactions_df.sort_values(['user_id', 'timestamp'])
    
    # Group by user
    user_sequences = interactions_df.groupby('user_id')['item_id'].apply(list)
    
    train_data = []
    test_data = []
    
    for user_id, seq in user_sequences.items():
        if len(seq) < 2:
            continue
        
        # Training: all but last
        train_seq = seq[:-1]
        train_target = seq[-1]
        train_data.append({
            'session_id': user_id,
            'item_id_seq': ' '.join(map(str, train_seq)),
            'item_id': train_target
        })
        
        # Test: full sequence
        test_seq = seq[:-1]
        test_target = seq[-1]
        test_data.append({
            'session_id': user_id,
            'item_id_seq': ' '.join(map(str, test_seq)),
            'item_id': test_target
        })
    
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    return train_df, test_df

# Usage
# interactions = pd.read_csv('raw_interactions.csv')
# train_df, test_df = create_sequential_data(interactions)
# train_df.to_csv('train.tsv', sep='\t', index=False)
# test_df.to_csv('test.tsv', sep='\t', index=False)
```

## Data Quality Checks

Before training, verify:
- [ ] Files exist: `train.tsv`, `test.tsv`
- [ ] Format is correct (TSV with 3 columns)
- [ ] Item IDs start from 1
- [ ] No missing values
- [ ] Sequences are not empty
- [ ] Test users exist in training data
- [ ] (Optional) Semantic embeddings match item count

## Troubleshooting

### "File not found" error
- Check file path: `datasets/sequential/mooc/train.tsv`
- Ensure files are in the correct directory

### "Index out of range" error
- Check if item IDs are consecutive (1, 2, 3, ...)
- Verify no item ID exceeds the maximum

### "Empty sequence" error
- Remove sequences with length < 1
- Check for null values in item_id_seq

### Poor performance
- Check data quality
- Verify train/test split is reasonable
- Ensure sufficient training data
- Check for data leakage

## License

Please respect the license of any dataset you use. Ensure you have the right to use and distribute the data.

## Contact

For dataset-related questions:
- Open an issue on GitHub
- Email: your.email@example.com
