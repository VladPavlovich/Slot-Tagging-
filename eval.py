import torch
from seqeval.metrics import f1_score

def id_to_tags(pred_ids, tag2id):
    inv = {v: k for k, v in tag2id.items()}
    result = []
    for seq in pred_ids:
        tags = []
        for tid in seq:
            if tid != 0:  # ignore PAD
                tags.append(inv[tid])
        result.append(tags)
    return result


def evaluate(model, X, Y, tag2id):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        logits = model(X.to(device))
        preds = torch.argmax(logits, dim=-1)

    preds = preds.cpu().tolist()
    true = Y.cpu().tolist()

    y_pred = id_to_tags(preds, tag2id)
    y_true = id_to_tags(true, tag2id)

    return f1_score(y_true, y_pred)
