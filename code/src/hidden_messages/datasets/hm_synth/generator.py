"""Deterministic HM-Synth examples with aligned counterfactual pairs."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass

MetadataValue = int | str | list[int] | list[str]


@dataclass(frozen=True, slots=True)
class HMSynthExample:
    example_id: str
    pair_id: str
    variant: str
    family: str
    question: str
    private_contexts: tuple[str, ...]
    designated_receiver: int
    answer: str
    metadata: dict[str, MetadataValue]

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, payload: object) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}-{hashlib.sha256(serialized.encode()).hexdigest()[:20]}"


def _names(rng: random.Random, count: int) -> list[str]:
    syllables = ("al", "bor", "cy", "dra", "el", "fyn", "gor", "hal", "io", "jun")
    pool = [
        f"{left}{right}{index}"
        for index in range(count * 2)
        for left, right in [
            (syllables[index % len(syllables)], syllables[(index * 7 + 3) % len(syllables)])
        ]
    ]
    rng.shuffle(pool)
    return pool[:count]


def generate_chain_lookup_pair(
    seed: int,
    *,
    num_agents: int = 2,
    hops: int = 2,
    distractors_per_agent: int = 2,
    answer_values: tuple[str, ...] | None = None,
) -> tuple[HMSynthExample, HMSynthExample]:
    """Return factual/counterfactual variants sharing receiver-local evidence."""

    if num_agents not in {2, 4}:
        raise ValueError("num_agents must be 2 or 4")
    if not 2 <= hops <= 6:
        raise ValueError("hops must be in [2, 6]")
    if distractors_per_agent < 0:
        raise ValueError("distractors_per_agent must be non-negative")
    if answer_values is not None and len(set(answer_values)) < 2:
        raise ValueError("answer_values must contain at least two distinct values")
    rng = random.Random(seed)
    entities = _names(rng, hops + 1 + num_agents * distractors_per_agent * 2)
    chain = entities[: hops + 1]
    if answer_values is None:
        code: int | str = rng.randint(10, 999)
        counterfactual_code: int | str = rng.randint(10, 999)
        while counterfactual_code == code:
            counterfactual_code = rng.randint(10, 999)
    else:
        code = rng.choice(answer_values)
        counterfactual_code = rng.choice(answer_values)
        while counterfactual_code == code:
            counterfactual_code = rng.choice(answer_values)

    facts_by_agent: list[list[str]] = [[] for _ in range(num_agents)]
    edge_owners: list[int] = []
    for hop in range(hops):
        owner = hop % num_agents
        edge_owners.append(owner)
        facts_by_agent[owner].append(f"{chain[hop]} is linked to {chain[hop + 1]}.")
    # The terminal code is remote from agent_0 by construction.
    terminal_owner = 1 if num_agents == 2 else (hops % (num_agents - 1)) + 1
    facts_by_agent[terminal_owner].append(f"{chain[-1]} has code {code}.")

    cursor = hops + 1
    for agent_id in range(num_agents):
        for _ in range(distractors_per_agent):
            left, right = entities[cursor], entities[cursor + 1]
            cursor += 2
            facts_by_agent[agent_id].append(f"{left} is linked to {right}.")
        rng.shuffle(facts_by_agent[agent_id])

    pair_payload: dict[str, MetadataValue] = {
        "family": "distributed_chain_lookup",
        "seed": seed,
        "num_agents": num_agents,
        "hops": hops,
        "distractors_per_agent": distractors_per_agent,
        "chain": chain,
        "edge_owners": edge_owners,
        "terminal_owner": terminal_owner,
        "answer_values": list(answer_values) if answer_values is not None else "integers_10_999",
    }
    pair_id = _stable_id("hm-chain-pair", pair_payload)
    base_contexts = tuple(" ".join(facts) for facts in facts_by_agent)
    answer_instruction = (
        "Return only the integer." if answer_values is None else "Return only the code."
    )
    question = f"What is the code associated with {chain[0]}? {answer_instruction}"

    factual = HMSynthExample(
        example_id=f"{pair_id}-a",
        pair_id=pair_id,
        variant="factual",
        family="distributed_chain_lookup",
        question=question,
        private_contexts=base_contexts,
        designated_receiver=0,
        answer=str(code),
        metadata={
            **pair_payload,
            "answer_code": code,
            "counterfactual_code": counterfactual_code,
        },
    )

    counterfactual_contexts = list(base_contexts)
    counterfactual_contexts[terminal_owner] = counterfactual_contexts[terminal_owner].replace(
        f"{chain[-1]} has code {code}.",
        f"{chain[-1]} has code {counterfactual_code}.",
    )
    counterfactual = HMSynthExample(
        example_id=f"{pair_id}-b",
        pair_id=pair_id,
        variant="counterfactual",
        family="distributed_chain_lookup",
        question=question,
        private_contexts=tuple(counterfactual_contexts),
        designated_receiver=0,
        answer=str(counterfactual_code),
        metadata={
            **pair_payload,
            "answer_code": counterfactual_code,
            "counterfactual_of": factual.example_id,
        },
    )
    return factual, counterfactual


def generate_modular_arithmetic_pair(
    seed: int,
    *,
    modulus: int = 97,
) -> tuple[HMSynthExample, HMSynthExample]:
    if modulus <= 7:
        raise ValueError("modulus must be greater than 7")
    rng = random.Random(seed)
    a, b, c, d = (rng.randrange(1, modulus) for _ in range(4))
    alt_d = rng.randrange(1, modulus)
    while alt_d == d:
        alt_d = rng.randrange(1, modulus)
    answer = ((a + b) * c - d) % modulus
    alt_answer = ((a + b) * c - alt_d) % modulus
    pair_payload: dict[str, MetadataValue] = {
        "family": "distributed_modular_arithmetic",
        "seed": seed,
        "modulus": modulus,
    }
    pair_id = _stable_id("hm-mod-pair", pair_payload)
    question = "Evaluate ((a + b) * c - d) mod p. Return only the integer."
    context_0 = f"a={a}; c={c}; p={modulus}."
    context_1 = f"b={b}; d={d}."
    common_metadata: dict[str, MetadataValue] = {
        **pair_payload,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "counterfactual_d": alt_d,
    }
    factual = HMSynthExample(
        example_id=f"{pair_id}-a",
        pair_id=pair_id,
        variant="factual",
        family="distributed_modular_arithmetic",
        question=question,
        private_contexts=(context_0, context_1),
        designated_receiver=0,
        answer=str(answer),
        metadata=common_metadata,
    )
    counterfactual = HMSynthExample(
        example_id=f"{pair_id}-b",
        pair_id=pair_id,
        variant="counterfactual",
        family="distributed_modular_arithmetic",
        question=question,
        private_contexts=(context_0, f"b={b}; d={alt_d}."),
        designated_receiver=0,
        answer=str(alt_answer),
        metadata={**common_metadata, "counterfactual_of": factual.example_id},
    )
    return factual, counterfactual


def generate_rendezvous_lookup_pair(
    seed: int,
    *,
    slot_values: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J"),
    answer_values: tuple[str, ...] = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"),
) -> tuple[HMSynthExample, HMSynthExample]:
    """Return a two-way lookup pair with no informative local-only agent view.

    Agent 0 knows which slot is requested but has no code table. Agent 1 has a balanced table but
    is not told which slot is requested. Union context resolves the answer. The counterfactual
    swaps the requested slot's code with another slot while preserving agent 0's context.
    """

    if len(slot_values) < 4:
        raise ValueError("rendezvous lookup requires at least four slots")
    if len(slot_values) != len(answer_values):
        raise ValueError("slot_values and answer_values must have equal length")
    if len(set(slot_values)) != len(slot_values):
        raise ValueError("slot_values must be unique")
    if len(set(answer_values)) != len(answer_values):
        raise ValueError("answer_values must be unique")

    rng = random.Random(seed)
    requested_index = rng.randrange(len(slot_values))
    requested_slot = slot_values[requested_index]
    factual_codes = list(answer_values)
    rng.shuffle(factual_codes)
    swap_index = rng.randrange(len(slot_values) - 1)
    if swap_index >= requested_index:
        swap_index += 1
    counterfactual_codes = factual_codes.copy()
    counterfactual_codes[requested_index], counterfactual_codes[swap_index] = (
        counterfactual_codes[swap_index],
        counterfactual_codes[requested_index],
    )
    table_order = list(range(len(slot_values)))
    rng.shuffle(table_order)

    def table_context(codes: list[str]) -> str:
        return " ".join(
            f"Slot {slot_values[index]} has code {codes[index]}." for index in table_order
        )

    receiver_context = f"The requested slot is {requested_slot}."
    factual_sender = table_context(factual_codes)
    counterfactual_sender = table_context(counterfactual_codes)
    pair_payload: dict[str, MetadataValue] = {
        "family": "distributed_rendezvous_lookup",
        "seed": seed,
        "slot_values": list(slot_values),
        "answer_values": list(answer_values),
        "requested_slot": requested_slot,
        "table_order": [slot_values[index] for index in table_order],
        "factual_codes": factual_codes,
        "counterfactual_codes": counterfactual_codes,
    }
    pair_id = _stable_id("hm-rendezvous-pair", pair_payload)
    question = "What code is stored in the requested slot? Return only the code."
    factual = HMSynthExample(
        example_id=f"{pair_id}-a",
        pair_id=pair_id,
        variant="factual",
        family="distributed_rendezvous_lookup",
        question=question,
        private_contexts=(receiver_context, factual_sender),
        designated_receiver=0,
        answer=factual_codes[requested_index],
        metadata={
            **pair_payload,
            "answer_code": factual_codes[requested_index],
            "counterfactual_code": counterfactual_codes[requested_index],
        },
    )
    counterfactual = HMSynthExample(
        example_id=f"{pair_id}-b",
        pair_id=pair_id,
        variant="counterfactual",
        family="distributed_rendezvous_lookup",
        question=question,
        private_contexts=(receiver_context, counterfactual_sender),
        designated_receiver=0,
        answer=counterfactual_codes[requested_index],
        metadata={
            **pair_payload,
            "answer_code": counterfactual_codes[requested_index],
            "counterfactual_of": factual.example_id,
        },
    )
    return factual, counterfactual
