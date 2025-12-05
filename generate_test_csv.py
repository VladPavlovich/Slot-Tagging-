import csv
from preprocess import prepare_test_dataset
import torch


def generate_submission(model, word2id, tag2id, input_path="test.csv", output_path="test_pred.csv"):
    device = next(model.parameters()).device
    
    # 1. Prepare test data
    X_test, test_lengths, ids = prepare_test_dataset(input_path, word2id)
    
    # Invert tag2id for decoding
    id2tag = {v: k for k, v in tag2id.items()}
    
    model.eval()
    predictions = []

    print(f"Generating predictions for {len(ids)} sentences...")

    with torch.no_grad():
        logits = model(X_test.to(device))
        preds = torch.argmax(logits, dim=-1)
        
        preds = preds.cpu().tolist()
        lengths = test_lengths.cpu().tolist()
        
        for i, (pred_seq, length) in enumerate(zip(preds, lengths)):
            # Get valid prediction length
            valid_preds = pred_seq[:length]
            
            # debugging check
            if length < 3: 
                print(f"Warning: ID {ids[i]} is very short ({length} tags). Check this row.")

            # Decode IDs to Strings
            raw_tags = [id2tag.get(tid, "O") for tid in valid_preds]
            
            cleaned_tags = []
            previous_tag = "O"

            # (Fixing the IOB Logic) ---
            for tag in raw_tags:
                # 1. Clean special tokens
                if tag in ["<PAD>", "<UNK>"]:
                    tag = "O"
                
                # 2. Fix Underscores to Hyphens (B_movie -> B-movie)
                # NOTE: We temporarily use hyphens to enforce IOB rules!
                tag = tag.replace("_", "-")

                # 3. Enforce valid IOB2 logic
                if tag.startswith("I-"):
                    type_label = tag.split("-")[1] # Get 'movie' from 'I-movie'
                    
                    # Check if previous tag supports this I-tag
                    valid_predecessors = [f"B-{type_label}", f"I-{type_label}"]
                    
                    if previous_tag not in valid_predecessors:
                        # Logic violation found! (e.g. O -> I-movie)
                        # Force 'I-' to become 'B-'
                        tag = f"B-{type_label}"

                cleaned_tags.append(tag)
                previous_tag = tag
            
            # --- FINAL STEP: Convert Hyphens back to Underscores ---
            final_tags = [t.replace('-', '_') for t in cleaned_tags]
            
            # Join with space
            tags_str = " ".join(final_tags)
            predictions.append((ids[i], tags_str))

    # Write to CSV
    with open(output_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "IOB Slot tags"]) 
        writer.writerows(predictions)
        
    print(f"Done. Predictions saved to {output_path}")