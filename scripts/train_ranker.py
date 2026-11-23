"""Train and evaluate a residual pairwise ranker over fixed DINOv2 candidates."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from audit_distribution import atomic_json, load_jsonl
from prepare_gallery import write_rows
from retrieval import HIT_KM, REVISION, feature_key

SEED = 7818


class Ranker(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.correction = nn.Sequential(nn.Linear(2 * dimension + 1, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, query, candidate):
        product = query * candidate
        cosine = product.sum(dim=-1, keepdim=True)
        delta = self.correction(torch.cat((product, (query - candidate).abs(), cosine), dim=-1))
        return (cosine + delta).squeeze(-1)


def load_features(folder, name, rows, device):
    with np.load(folder / f"{name}_features.npz", allow_pickle=False) as z:
        if str(z["feature_key"]) != feature_key(rows) or str(z["revision"]) != REVISION:
            raise ValueError(f"Embedding cache does not match {name} manifest or encoder revision")
        features = z["features"]
    if features.shape != (len(rows), 384) or not np.isfinite(features).all():
        raise ValueError(f"Invalid {name} embedding matrix")
    norms = np.linalg.norm(features.astype(np.float32), axis=-1)
    if np.any(norms == 0):
        raise ValueError(f"Zero embedding norm in {name}")
    return F.normalize(torch.as_tensor(features, dtype=torch.float32, device=device), dim=-1)


def load_candidates(folder, name, queries, gallery_size):
    rows = load_jsonl(folder / f"{name}_candidates.jsonl")
    if [r["query_id"] for r in rows] != [r["id"] for r in queries]:
        raise ValueError(f"Wrong candidate order for {name}")
    indices = np.asarray([r["gallery_indices"] for r in rows], dtype=np.int64)
    distances = np.asarray([r["distance_km"] for r in rows], dtype=np.float64)
    if (indices.ndim != 2 or indices.shape != distances.shape or indices.shape[1] < 1 or
            indices.min() < 0 or indices.max() >= gallery_size or not np.isfinite(distances).all()):
        raise ValueError(f"Invalid candidate arrays for {name}")
    if any(len(set(r["gallery_indices"])) != indices.shape[1] for r in rows):
        raise ValueError(f"Duplicate candidates in {name}")
    return indices, distances


def load_pairs(folder, indices, distances):
    pairs = np.load(folder / "train_pairs.npy", allow_pickle=False)
    if pairs.ndim != 2 or pairs.shape[1] != 3 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("Expected integer pair rows with query, positive and negative indices")
    if len(pairs) and (pairs.min() < 0 or pairs[:, 0].max() >= len(indices)):
        raise ValueError("Pair query index out of bounds")
    for query, positive, negative in pairs:
        p = np.flatnonzero(indices[query] == positive)
        n = np.flatnonzero(indices[query] == negative)
        if len(p) != 1 or len(n) != 1 or distances[query, p[0]] > HIT_KM or distances[query, n[0]] <= HIT_KM:
            raise ValueError(f"Pair label or retrieval membership mismatch for query {query}")
    return pairs


def choose(model, queries, references, indices):
    device = references.device
    picks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(queries), 128):
            rows = torch.as_tensor(indices[start:start + 128], device=device)
            q = queries[start:start + len(rows), None, :]
            picks.append(model(q, references[rows]).argmax(dim=1).cpu().numpy())
    return np.concatenate(picks)


def metrics(distance):
    return dict(n=len(distance), hits_25=int((distance <= HIT_KM).sum()),
                hit_rate_25=float((distance <= HIT_KM).mean()), hits_200=int((distance <= 200).sum()),
                median_km=float(np.median(distance)))


def train(folder, *, device, epochs, batch_size, min_queries):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    gallery = load_jsonl(folder / "gallery_images.jsonl")
    train_rows = load_jsonl(folder / "train_images.jsonl")
    val_rows = load_jsonl(folder / "validation_images.jsonl")
    train_indices, train_distances = load_candidates(folder, "train", train_rows, len(gallery))
    all_pairs = load_pairs(folder, train_indices, train_distances)
    pair_query_count = len(np.unique(all_pairs[:, 0]))
    if pair_query_count < min_queries:
        status = dict(status="insufficient_pairs", pair_queries=pair_query_count, target=min_queries,
                      train_queries=len(train_rows), top100_hits=int((train_distances.min(axis=1) <= HIT_KM).sum()))
        atomic_json(folder / "ranker_summary.json", status)
        print(json.dumps(status, indent=2), flush=True)
        return

    references = load_features(folder, "gallery", gallery, device)
    training_features = load_features(folder, "train", train_rows, device)
    validation_features = load_features(folder, "validation", val_rows, device)
    val_indices, val_distances = load_candidates(folder, "validation", val_rows, len(gallery))
    tune = rng.permutation(len(train_rows))[:max(1, len(train_rows) // 10)]
    pairs = all_pairs[~np.isin(all_pairs[:, 0], tune)]
    if not len(pairs):
        raise ValueError("No usable pairs left in the optimization split")
    query_counts = np.bincount(pairs[:, 0], minlength=len(train_rows))
    sample_weights = len(pairs) / (np.count_nonzero(query_counts) * query_counts[pairs[:, 0]])
    weights = torch.as_tensor(sample_weights, dtype=torch.float32, device=device)
    pairs = torch.as_tensor(pairs, dtype=torch.long, device=device)
    tune_features = training_features[torch.as_tensor(tune, device=device)]
    tune_indices, tune_distances = train_indices[tune], train_distances[tune]
    tune_rows = np.arange(len(tune))

    model = Ranker(references.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    initial = metrics(tune_distances[:, 0])
    best = (initial["hits_25"], -initial["median_km"])
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history = [dict(epoch=0, **initial)]
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        for batch in torch.randperm(len(pairs), device=device).split(batch_size):
            q, p, n = pairs[batch].unbind(dim=1)
            q_features = training_features[q]
            positive = model(q_features, references[p])
            negative = model(q_features, references[n])
            loss = (F.softplus(negative - positive) * weights[batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(batch)
        picks = choose(model, tune_features, references, tune_indices)
        score = metrics(tune_distances[tune_rows, picks])
        row = dict(epoch=epoch, loss=float(loss_sum / len(pairs)), **score)
        history.append(row)
        print(json.dumps(row), flush=True)
        key = (score["hits_25"], -score["median_km"])
        if key > best:
            best, best_state, best_epoch = key, copy.deepcopy(model.state_dict()), epoch
        if epoch - best_epoch >= 3:
            break

    model.load_state_dict(best_state)
    picks = (choose(model, validation_features, references, val_indices) if best_epoch
             else np.zeros(len(val_rows), dtype=np.int64))
    q_indices = np.arange(len(val_rows))
    baseline_km, ranker_km = val_distances[:, 0], val_distances[q_indices, picks]
    improved = int(((baseline_km > HIT_KM) & (ranker_km <= HIT_KM)).sum())
    regressed = int(((baseline_km <= HIT_KM) & (ranker_km > HIT_KM)).sum())
    summary = dict(status="trained", seed=SEED, lr=3e-4, batch_size=batch_size, best_epoch=best_epoch,
                   pair_queries=pair_query_count, optimization_queries=int(np.count_nonzero(query_counts)),
                   optimization_pairs=len(pairs), tuning_queries=len(tune), validation_queries=len(val_rows),
                   validation_top100_hits=int((val_distances.min(axis=1) <= HIT_KM).sum()),
                   baseline=metrics(baseline_km), reranked=metrics(ranker_km), improved=improved, regressed=regressed,
                   tuning_history=history)
    write_rows(folder / "ranker_predictions.jsonl", (
        dict(query_id=q["id"], cosine_id=gallery[val_indices[i, 0]]["id"],
             ranker_id=gallery[val_indices[i, picks[i]]]["id"], cosine_km=float(baseline_km[i]),
             ranker_km=float(ranker_km[i]), ranker_original_rank=int(picks[i] + 1),
             guess_latitude=gallery[val_indices[i, picks[i]]]["latitude"],
             guess_longitude=gallery[val_indices[i, picks[i]]]["longitude"])
        for i, q in enumerate(val_rows)
    ))
    checkpoint = folder / "ranker.pt"
    partial = checkpoint.with_suffix(".part")
    torch.save(dict(state_dict={k: v.cpu() for k, v in best_state.items()}, dimension=references.shape[1],
                    encoder_revision=REVISION, seed=SEED), partial)
    partial.replace(checkpoint)
    atomic_json(folder / "ranker_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-queries", type=int, default=2000)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.min_queries) < 1:
        parser.error("Epochs, batch size, and minimum query count must be positive")
    train(args.folder, device=args.device, epochs=args.epochs, batch_size=args.batch_size, min_queries=args.min_queries)


if __name__ == "__main__":
    main()
