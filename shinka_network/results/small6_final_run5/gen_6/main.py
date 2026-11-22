"""Surrogate QUBO terms for QOBLIB 08 network design. ShinkaEvolve edits the EVOLVE block."""

import time

# EVOLVE-BLOCK-START
import numpy as np


def surrogate_terms(demand):
    """Distill heuristic congestion scores into a quadratic star model."""
    traffic = np.asarray(demand, dtype=float).copy()
    if traffic.ndim != 2 or traffic.shape[0] != traffic.shape[1]:
        raise ValueError("demand must be a square matrix")

    n = traffic.shape[0]
    traffic = np.nan_to_num(
        traffic, nan=0.0, posinf=0.0, neginf=0.0
    )
    traffic = np.maximum(traffic, 0.0)
    np.fill_diagonal(traffic, 0.0)

    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    maximum = float(traffic.max()) if traffic.size else 0.0
    if maximum <= 0.0:
        return {edge: 0.0 for edge in edges}, {}, 2.0

    traffic /= maximum
    nodes = np.arange(n, dtype=np.int32)
    columns = nodes.copy()

    # Feature numbering: links, outgoing stars, incoming stars.
    link_id = np.full((n, n), -1, dtype=np.int32)
    for index, (i, j) in enumerate(edges):
        link_id[i, j] = index
    number_links = len(edges)

    star_id = np.full((n, n, n), -1, dtype=np.int32)
    next_id = number_links
    neighbors = []

    for center in range(n):
        other = np.array(
            [j for j in range(n) if j != center], dtype=np.int32
        )
        neighbors.append(other)
        for a in range(len(other)):
            j = int(other[a])
            for b in range(a + 1, len(other)):
                k = int(other[b])
                star_id[center, j, k] = next_id
                star_id[center, k, j] = next_id
                next_id += 1

    number_stars = next_id - number_links
    number_features = number_links + 2 * number_stars + 1
    reciprocal_id = number_features - 1

    # A reproducible integer schedule; no global random state is touched.
    state = (0x9E3779B9 ^ (n * 104729)) & 0xFFFFFFFF

    def integer(limit):
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return ((state >> 8) * limit) >> 24

    def unit():
        return integer(1 << 24) / float(1 << 24)

    def permutation():
        order = np.arange(n, dtype=np.int32)
        for k in range(n - 1, 0, -1):
            j = integer(k + 1)
            order[k], order[j] = order[j], order[k]
        return order

    def cycle():
        order = permutation()
        successor = np.empty(n, dtype=np.int32)
        successor[order] = np.roll(order, -1)
        return successor

    def shortest_distances(p, q, cost0=None, cost1=None):
        distance = np.full((n, n), 1.0e12)
        np.fill_diagonal(distance, 0.0)

        for _ in range(n - 1):
            if cost0 is None:
                proposal = 1.0 + np.minimum(distance[p], distance[q])
            else:
                proposal = np.minimum(
                    cost0[:, None] + distance[p],
                    cost1[:, None] + distance[q],
                )
            updated = np.minimum(distance, proposal)
            if np.array_equal(updated, distance):
                break
            distance = updated
        return distance

    mixture = np.array([0.0, 0.25, 0.5, 0.75, 1.0])[:, None]
    top_count = max(2, n // 3)

    def route_score(p, q):
        """Two feasible routing policies and their best convex mixture."""
        invp = np.empty(n, dtype=np.int32)
        invq = np.empty(n, dtype=np.int32)
        invp[p] = nodes
        invq[q] = nodes

        distance = shortest_distances(p, q)
        if float(distance.max()) > n:
            return None

        # First policy: equal splitting over all shortest next hops.
        eligible0 = distance[p] + 1.0 == distance
        eligible1 = distance[q] + 1.0 == distance
        denominator = np.maximum(
            eligible0.astype(float) + eligible1, 1.0
        )
        fraction0 = eligible0 / denominator

        residual = traffic.copy()
        load0 = np.zeros(n)
        load1 = np.zeros(n)

        for level in range(int(distance.max()), 0, -1):
            sending = residual * (distance == level)
            flow0 = sending * fraction0
            flow1 = sending - flow0

            load0 += flow0.sum(axis=1)
            load1 += flow1.sum(axis=1)
            residual += flow0[invp] + flow1[invq]

        first = np.concatenate((load0, load1))

        # Second policy: use congestion prices to discover alternative routes.
        mean_load = max(float(first.mean()), 1.0e-12)
        price0 = 1.0 + 0.75 * np.minimum(load0 / mean_load, 4.0) ** 2
        price1 = 1.0 + 0.75 * np.minimum(load1 / mean_load, 4.0) ** 2
        priced_distance = shortest_distances(p, q, price0, price1)

        continuation0 = priced_distance[p]
        continuation1 = priced_distance[q]
        length0 = price0[:, None] + continuation0
        length1 = price1[:, None] + continuation1

        # Strictly decreasing distances give an acyclic routing policy.
        allowed0 = continuation0 < priced_distance - 1.0e-10
        allowed1 = continuation1 < priced_distance - 1.0e-10

        difference = np.clip((length0 - length1) / 0.8, -35.0, 35.0)
        probability0 = 1.0 / (1.0 + np.exp(difference))
        probability0 = np.where(
            ~allowed0, 0.0, np.where(~allowed1, 1.0, probability0)
        )

        order = np.argsort(-priced_distance, axis=0, kind="stable")
        residual = traffic.copy()
        second0 = np.zeros(n)
        second1 = np.zeros(n)

        for rank in range(n - 1):
            current = order[rank]
            amount = residual[current, columns]
            flow0 = amount * probability0[current, columns]
            flow1 = amount - flow0

            second0 += np.bincount(
                current, weights=flow0, minlength=n
            )
            second1 += np.bincount(
                current, weights=flow1, minlength=n
            )
            residual[p[current], columns] += flow0
            residual[q[current], columns] += flow1

        second = np.concatenate((second0, second1))
        candidates = first[None, :] + mixture * (second - first)[None, :]

        largest = candidates.max(axis=1)
        upper = np.partition(
            candidates, 2 * n - top_count, axis=1
        )[:, -top_count:].mean(axis=1)

        # A smoothed peak is less noisy than the single largest load.
        scores = (
            0.70 * largest
            + 0.25 * upper
            + 0.05 * candidates.mean(axis=1)
        )
        best = int(np.argmin(scores))
        return float(scores[best]), candidates[best].reshape(2, n)

    feature_rows = []
    reciprocal_counts = []
    observed_scores = []
    cache = {}

    def evaluate(p, q):
        outgoing_features = star_id[nodes, p, q]
        key = outgoing_features.tobytes()

        if key in cache:
            cached = cache[key]
            if cached is None:
                return None
            score, old_p, old_load = cached
            same = p == old_p
            load = np.where(same[None, :], old_load, old_load[::-1])
            return score, load

        result = route_score(p, q)
        if result is None:
            cache[key] = None
            return None

        score, load = result
        invp = np.empty(n, dtype=np.int32)
        invq = np.empty(n, dtype=np.int32)
        invp[p] = nodes
        invq[q] = nodes

        row = np.concatenate((
            link_id[nodes, p],
            link_id[nodes, q],
            outgoing_features,
            star_id[nodes, invp, invq] + number_stars,
        ))
        reciprocal_count = 0.5 * (
            np.count_nonzero((p[p] == nodes) | (q[p] == nodes))
            + np.count_nonzero((p[q] == nodes) | (q[q] == nodes))
        )

        feature_rows.append(row)
        reciprocal_counts.append(float(reciprocal_count))
        observed_scores.append(score)
        cache[key] = (score, p.copy(), load.copy())
        return score, load

    # Generate a demand-specific training set with feasible local exchanges.
    elites = []
    number_starts = 6
    number_steps = 150 + 4 * n

    for _ in range(number_starts):
        p = cycle()
        q = cycle()
        for attempt in range(128):
            if np.all(p != q):
                break
            q = cycle()
        if np.any(p == q):
            q = np.empty(n, dtype=np.int32)
            q[p] = nodes  # Reverse the Hamiltonian cycle.

        current_score, current_load = evaluate(p, q)
        best_score = current_score
        best_p, best_q = p.copy(), q.copy()
        ranked_links = np.argsort(current_load.ravel())[::-1]

        for step in range(number_steps):
            if step % 3:
                selected = int(
                    ranked_links[integer(max(2, n // 2))]
                )
                color, a = divmod(selected, n)
            else:
                color, a = integer(2), integer(n)

            b = integer(n - 1)
            if b >= a:
                b += 1

            selected = p if color == 0 else q
            other = q if color == 0 else p
            destination_a = int(selected[a])
            destination_b = int(selected[b])

            if (
                destination_b == a
                or destination_a == b
                or destination_b == other[a]
                or destination_a == other[b]
            ):
                continue

            changed = selected.copy()
            changed[a], changed[b] = destination_b, destination_a
            trial_p = changed if color == 0 else p
            trial_q = q if color == 0 else changed

            result = evaluate(trial_p, trial_q)
            if result is None:
                continue
            trial_score, trial_load = result

            if trial_score < best_score:
                best_score = trial_score
                best_p, best_q = trial_p.copy(), trial_q.copy()

            progress = step / max(1, number_steps - 1)
            tolerance = 0.028 * (1.0 - progress) ** 2 + 0.0007
            permitted = tolerance * (0.4 + 1.6 * unit())

            if np.log(trial_score / current_score) <= permitted:
                p, q = trial_p, trial_q
                current_score, current_load = trial_score, trial_load
                ranked_links = np.argsort(current_load.ravel())[::-1]

        elites.append((best_score, best_p, best_q))

    # Fit a regularized sparse quadratic predictor, including an intercept
    # implicitly by weighted centering. No dense design matrix is needed.
    indices = np.asarray(feature_rows, dtype=np.int32)
    reciprocals = np.asarray(reciprocal_counts)
    scores = np.asarray(observed_scores)
    targets = np.log(np.maximum(scores, 1.0e-15))
    width = indices.shape[1]
    flat_indices = indices.ravel()

    weights = np.maximum(
        0.10, np.exp(-(targets - targets.min()) / 0.15)
    )
    weights /= weights.mean()
    weight_sum = float(weights.sum())

    prior = np.zeros(number_features)
    support = 0.8 * traffic + 0.2 * np.sqrt(traffic)
    support_sum = max(float(support.sum()), 1.0e-15)
    for k, (i, j) in enumerate(edges):
        prior[k] = -0.90 * support[i, j] / support_sum
    prior[reciprocal_id] = 0.08 / n

    regularizer = np.full(number_features, 2.0 * n)
    regularizer[:number_links] = 0.6 * n
    regularizer[reciprocal_id] = 1.0

    def predict(coefficients):
        return (
            coefficients[indices].sum(axis=1)
            + coefficients[reciprocal_id] * reciprocals
        )

    def transpose(values):
        result = np.bincount(
            flat_indices,
            weights=np.repeat(values, width),
            minlength=number_features,
        )
        result[reciprocal_id] += float(values @ reciprocals)
        return result

    counts = transpose(weights)
    feature_mean = counts / weight_sum

    residual_targets = targets - predict(prior)
    residual_targets -= float(weights @ residual_targets) / weight_sum
    right_hand_side = transpose(weights * residual_targets)

    diagonal = counts.copy()
    diagonal[reciprocal_id] = float(weights @ (reciprocals ** 2))
    diagonal -= weight_sum * feature_mean ** 2
    diagonal = np.maximum(diagonal, 0.0) + regularizer

    def multiply(vector):
        fitted = predict(vector) - float(feature_mean @ vector)
        return transpose(weights * fitted) + regularizer * vector

    # Preconditioned conjugate gradients for ordinary ridge regression.
    correction = np.zeros(number_features)
    residual = right_hand_side.copy()
    preconditioned = residual / diagonal
    direction = preconditioned.copy()
    product = float(residual @ preconditioned)
    initial_norm = max(float(np.linalg.norm(residual)), 1.0e-15)

    for _ in range(55):
        if product <= 1.0e-24:
            break
        image = multiply(direction)
        denominator = float(direction @ image)
        if denominator <= 1.0e-24:
            break

        step_size = product / denominator
        correction += step_size * direction
        residual -= step_size * image

        if float(np.linalg.norm(residual)) < 1.0e-6 * initial_norm:
            break

        preconditioned = residual / diagonal
        new_product = float(residual @ preconditioned)
        direction = preconditioned + (new_product / product) * direction
        product = new_product

    coefficients = prior + correction

    # A soft elite prior helps generation where regression must extrapolate.
    elite_scores = np.array([entry[0] for entry in elites])
    elite_weights = np.exp(
        -np.log(elite_scores / elite_scores.min()) / 0.045
    )
    elite_weights /= elite_weights.sum()

    for weight, (_, p, q) in zip(elite_weights, elites):
        invp = np.empty(n, dtype=np.int32)
        invq = np.empty(n, dtype=np.int32)
        invp[p] = nodes
        invq[q] = nodes

        coefficients[link_id[nodes, p]] -= 0.12 * weight / (2 * n)
        coefficients[link_id[nodes, q]] -= 0.12 * weight / (2 * n)
        coefficients[star_id[nodes, p, q]] -= (
            0.035 * weight / (2 * n)
        )
        coefficients[
            star_id[nodes, invp, invq] + number_stars
        ] -= 0.035 * weight / (2 * n)

    linear_matrix = np.zeros((n, n))
    for k, (i, j) in enumerate(edges):
        linear_matrix[i, j] = coefficients[k]

    quadratic = {}

    # Gauge away additive star components. On degree-two networks this
    # transformation is exact, and it reduces unnecessary coefficient range.
    for center in range(n):
        other = neighbors[center]
        count = len(other)
        ids = star_id[center][np.ix_(other, other)]

        for incoming in (False, True):
            offset = number_stars if incoming else 0
            block = coefficients[ids + offset].copy()
            np.fill_diagonal(block, 0.0)

            row_sums = block.sum(axis=1)
            additive = (
                row_sums - row_sums.sum() / (2.0 * (count - 1))
            ) / (count - 2)

            block -= additive[:, None] + additive[None, :]
            np.fill_diagonal(block, 0.0)

            if incoming:
                linear_matrix[other, center] += additive
            else:
                linear_matrix[center, other] += additive

            for a in range(count):
                j = int(other[a])
                for b in range(a + 1, count):
                    k = int(other[b])
                    value = float(block[a, b])
                    if abs(value) <= 1.0e-12:
                        continue
                    if incoming:
                        first, second = (j, center), (k, center)
                    else:
                        first, second = (center, j), (center, k)
                    key = (
                        (first, second)
                        if first < second else (second, first)
                    )
                    quadratic[key] = value

    reciprocal_weight = max(0.0, float(coefficients[reciprocal_id]))
    if reciprocal_weight > 1.0e-12:
        for i in range(n):
            for j in range(i + 1, n):
                quadratic[((i, j), (j, i))] = reciprocal_weight

    # Node potentials are constant over feasible networks. Remove them
    # before the evaluator's maximum-coefficient normalization.
    for _ in range(12):
        linear_matrix -= (
            linear_matrix.sum(axis=1) / (n - 1)
        )[:, None]
        np.fill_diagonal(linear_matrix, 0.0)
        linear_matrix -= (
            linear_matrix.sum(axis=0) / (n - 1)
        )[None, :]
        np.fill_diagonal(linear_matrix, 0.0)

    linear = {
        (i, j): float(linear_matrix[i, j])
        for i, j in edges
    }
    return linear, quadratic, 2.0


# EVOLVE-BLOCK-END


def run_surrogate(demand):
    """Fixed entry point called by the evaluator, once per demand matrix."""
    demand = np.array(demand, dtype=float)
    start = time.perf_counter()
    linear, quadratic, penalty_weight = surrogate_terms(demand.copy())
    return {
        "n": len(demand),
        "linear": linear,
        "quadratic": quadratic,
        "penalty_weight": penalty_weight,
        "seconds": time.perf_counter() - start,
    }
