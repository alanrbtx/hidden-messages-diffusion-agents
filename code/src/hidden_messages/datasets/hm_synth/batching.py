"""Deterministic pair-preserving batches for causal communication controls."""

from __future__ import annotations

from collections import Counter, defaultdict

from hidden_messages.datasets.hm_synth.generator import HMSynthExample


def pair_preserving_control_batches(
    examples: list[HMSynthExample],
    *,
    batch_examples: int,
) -> list[list[HMSynthExample]]:
    """Rebatch intact counterfactual pairs so different-slot permutations exist.

    Each rendezvous pair has one requested slot repeated across its factual and
    counterfactual examples. A different-slot permutation exists exactly when no slot
    occupies more than half of a batch. The returned schedule changes only execution
    order: it neither drops nor duplicates an example, and every pair remains adjacent.
    """

    if batch_examples <= 0 or batch_examples % 2:
        raise ValueError("batch_examples must be a positive even number")
    if len(examples) % batch_examples:
        raise ValueError("control evaluation requires full pair-aligned batches")
    pairs_per_batch = batch_examples // 2
    if pairs_per_batch < 2:
        raise ValueError("a control batch must contain at least two counterfactual pairs")

    pair_records: list[tuple[int, str, tuple[HMSynthExample, HMSynthExample]]] = []
    for pair_index, start in enumerate(range(0, len(examples), 2)):
        factual, counterfactual = examples[start : start + 2]
        if factual.pair_id != counterfactual.pair_id:
            raise ValueError("counterfactual pair adjacency was broken")
        if {factual.variant, counterfactual.variant} != {"factual", "counterfactual"}:
            raise ValueError("each pair must contain factual and counterfactual variants")
        factual_slot = str(factual.metadata["requested_slot"])
        counterfactual_slot = str(counterfactual.metadata["requested_slot"])
        if factual_slot != counterfactual_slot:
            raise ValueError("requested slot changed inside a counterfactual pair")
        if factual.answer == counterfactual.answer:
            raise ValueError("answer did not change inside a counterfactual pair")
        pair_records.append((pair_index, factual_slot, (factual, counterfactual)))

    batch_count = len(examples) // batch_examples
    max_same_slot_pairs = pairs_per_batch // 2
    by_slot: dict[str, list[tuple[int, str, tuple[HMSynthExample, HMSynthExample]]]] = defaultdict(
        list
    )
    for record in pair_records:
        by_slot[record[1]].append(record)
    for slot, records in by_slot.items():
        if len(records) > batch_count * max_same_slot_pairs:
            raise ValueError(f"slot {slot!r} is too frequent for different-slot control batches")

    bins: list[list[tuple[int, str, tuple[HMSynthExample, HMSynthExample]]]] = [
        [] for _ in range(batch_count)
    ]
    slot_counts: list[Counter[str]] = [Counter() for _ in range(batch_count)]
    groups = sorted(by_slot.items(), key=lambda item: (-len(item[1]), item[0]))
    for slot, records in groups:
        for record in records:
            candidates = [
                index
                for index in range(batch_count)
                if len(bins[index]) < pairs_per_batch
                and slot_counts[index][slot] < max_same_slot_pairs
            ]
            if not candidates:
                raise RuntimeError("deterministic control rebatching could not find a valid bin")
            selected = min(
                candidates,
                key=lambda index: (
                    slot_counts[index][slot],
                    len(bins[index]),
                    index,
                ),
            )
            bins[selected].append(record)
            slot_counts[selected][slot] += 1

    batches: list[list[HMSynthExample]] = []
    observed_pair_indices: list[int] = []
    for records in bins:
        if len(records) != pairs_per_batch:
            raise RuntimeError("deterministic control rebatching produced a partial batch")
        records.sort(key=lambda record: record[0])
        if max(Counter(record[1] for record in records).values()) > max_same_slot_pairs:
            raise AssertionError("different-slot control remains infeasible after rebatching")
        answer_counts = Counter(example.answer for _, _, pair in records for example in pair)
        if max(answer_counts.values()) > batch_examples // 2:
            raise AssertionError("different-target control is infeasible after rebatching")
        batch: list[HMSynthExample] = []
        for original_index, _, pair in records:
            observed_pair_indices.append(original_index)
            batch.extend(pair)
        batches.append(batch)

    if sorted(observed_pair_indices) != list(range(len(pair_records))):
        raise AssertionError("control rebatching dropped or duplicated a pair")
    return batches
