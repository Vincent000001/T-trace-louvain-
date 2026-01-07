import csv
import json
import math
import os
import random
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    IMAGEIO_AVAILABLE = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def modularity(G: nx.Graph, partition: dict) -> float:
    """Compute modularity Q for a given partition."""
    m = G.size(weight="weight")
    if m == 0:
        return 0.0

    comm_degree = defaultdict(float)
    comm_in = defaultdict(float)

    for u, v, w in G.edges(data="weight", default=1.0):
        cu = partition[u]
        cv = partition[v]
        if cu == cv:
            comm_in[cu] += w
        comm_degree[cu] += G.degree(u, weight="weight")
        comm_degree[cv] += G.degree(v, weight="weight")

    Q = 0.0
    for c in set(partition.values()):
        in_w = comm_in[c]
        tot = comm_degree[c]
        Q += (in_w / (2.0 * m)) - (tot / (2.0 * m)) ** 2
    return Q


def compute_delta_Q(G: nx.Graph, partition: dict, node, target_comm) -> float:
    """Brute-force delta Q for moving node to target_comm."""
    if partition[node] == target_comm:
        return 0.0
    original_comm = partition[node]
    old_Q = modularity(G, partition)
    partition[node] = target_comm
    new_Q = modularity(G, partition)
    partition[node] = original_comm
    return new_Q - old_Q


def local_moving(
    G: nx.Graph,
    partition: dict,
    trace_log: list,
    on_move,
) -> dict:
    nodes = list(G.nodes())
    improved = True

    while improved:
        improved = False
        random.shuffle(nodes)
        for node in nodes:
            current_comm = partition[node]
            neighbor_comms = {partition[nb] for nb in G.neighbors(node)}
            neighbor_comms.add(current_comm)

            for target_comm in neighbor_comms:
                delta = compute_delta_Q(G, partition, node, target_comm)
                if delta > 0.0:
                    partition[node] = target_comm
                    trace_log.append(
                        f"move: node={node}, from={current_comm}, to={target_comm}, ΔQ={delta:.6f}"
                    )
                    on_move(node, current_comm, target_comm, delta, partition)
                    improved = True
                    current_comm = target_comm
    return partition


def local_moving_queue(
    G: nx.Graph,
    partition: dict,
    trace_log: list,
    on_move,
    theta: float,
) -> dict:
    queue = list(G.nodes())
    in_queue = {node: True for node in queue}

    while queue:
        node = queue.pop(0)
        in_queue[node] = False
        current_comm = partition[node]
        neighbor_comms = {partition[nb] for nb in G.neighbors(node)}
        neighbor_comms.add(current_comm)

        deltas = []
        for target_comm in neighbor_comms:
            delta = compute_delta_Q(G, partition, node, target_comm)
            if delta > 0.0:
                deltas.append((target_comm, delta))

        if not deltas:
            continue

        if theta and theta > 0:
            weights = [math.exp(delta / theta) for _, delta in deltas]
            target_comm, delta = random.choices(deltas, weights=weights, k=1)[0]
        else:
            target_comm, delta = max(deltas, key=lambda item: item[1])

        if target_comm != current_comm:
            partition[node] = target_comm
            trace_log.append(
                f"move: node={node}, from={current_comm}, to={target_comm}, ΔQ={delta:.6f}"
            )
            on_move(node, current_comm, target_comm, delta, partition)
            for nb in G.neighbors(node):
                if not in_queue.get(nb, False):
                    queue.append(nb)
                    in_queue[nb] = True

    return partition


def aggregate_graph(G: nx.Graph, partition: dict) -> nx.Graph:
    newG = nx.Graph()
    comm_nodes = defaultdict(list)
    for node, comm in partition.items():
        comm_nodes[comm].append(node)

    for comm in comm_nodes:
        newG.add_node(comm)

    for u, v, w in G.edges(data="weight", default=1.0):
        cu = partition[u]
        cv = partition[v]
        if newG.has_edge(cu, cv):
            newG[cu][cv]["weight"] += w
        else:
            newG.add_edge(cu, cv, weight=w)
    return newG


def expand_partition(partition: dict, node_mapping: dict) -> dict:
    expanded = {}
    for cg_node, comm in partition.items():
        for orig in node_mapping[cg_node]:
            expanded[orig] = comm
    return expanded


def count_disconnected_communities(G: nx.Graph, partition: dict) -> int:
    comm_nodes = defaultdict(list)
    for node, comm in partition.items():
        comm_nodes[comm].append(node)
    count = 0
    for nodes in comm_nodes.values():
        subgraph = G.subgraph(nodes)
        if subgraph.number_of_nodes() > 1 and not is_connected_subgraph(G, nodes):
            count += 1
    return count


def refine_partitions(G: nx.Graph, partition: dict):
    triggered = 0
    before = count_disconnected_communities(G, partition)
    if before == 0:
        return partition, triggered, before, before

    triggered = 1
    new_partition = partition.copy()
    next_comm = max(new_partition.values(), default=-1) + 1
    comm_nodes = defaultdict(list)
    for node, comm in partition.items():
        comm_nodes[comm].append(node)

    for comm, nodes in comm_nodes.items():
        subgraph = G.subgraph(nodes)
        if subgraph.number_of_nodes() > 1 and not is_connected_subgraph(G, nodes):
            components = list(nx.connected_components(subgraph))
            for component in components[1:]:
                for node in component:
                    new_partition[node] = next_comm
                next_comm += 1

    after = count_disconnected_communities(G, new_partition)
    return new_partition, triggered, before, after


def louvain_with_trace(G: nx.Graph, seed=42):
    set_seed(seed)
    trace_log = []
    moves = []
    hierarchy = []
    step_counter = {"step": 0}

    current_G = G.copy()
    node_mapping = {n: {n} for n in current_G.nodes()}
    history = []
    modularity_history = []

    partition = {n: i for i, n in enumerate(current_G.nodes())}
    expanded = expand_partition(partition, node_mapping)
    history.append(expanded)
    modularity_history.append(modularity(G, expanded))

    def on_move(node, from_comm, to_comm, delta, current_partition):
        step_counter["step"] += 1
        expanded_partition = expand_partition(current_partition, node_mapping)
        history.append(expanded_partition)
        q_total = modularity(G, expanded_partition)
        modularity_history.append(q_total)
        moves.append(
            {
                "step": step_counter["step"],
                "algorithm": "standard",
                "node": node,
                "from": from_comm,
                "to": to_comm,
                "delta_q": delta,
                "q_total": q_total,
            }
        )

    while True:
        partition = local_moving(current_G, partition, trace_log, on_move)

        newG = aggregate_graph(current_G, partition)
        if newG.number_of_nodes() == current_G.number_of_nodes():
            break

        hierarchy.append(
            {
                "level": len(hierarchy),
                "mapping": _community_merge_mapping(partition),
            }
        )

        new_mapping = {}
        for comm in newG.nodes():
            members = set()
            for old_node, old_comm in partition.items():
                if old_comm == comm:
                    members |= node_mapping[old_node]
            new_mapping[comm] = members

        current_G = newG
        node_mapping = new_mapping
        partition = {n: i for i, n in enumerate(current_G.nodes())}
        expanded = expand_partition(partition, node_mapping)
        history.append(expanded)
        modularity_history.append(modularity(G, expanded))

    final_partition = history[-1]
    return final_partition, history, modularity_history, trace_log, moves, hierarchy


def enhanced_louvain_with_trace(G: nx.Graph, seed=42, theta: float = 0.0):
    set_seed(seed)
    trace_log = []
    moves = []
    hierarchy = []
    refine_stats = {
        "triggered": 0,
        "disconnected_before": 0,
        "disconnected_after": 0,
    }
    step_counter = {"step": 0}

    current_G = G.copy()
    node_mapping = {n: {n} for n in current_G.nodes()}
    history = []
    modularity_history = []

    partition = {n: i for i, n in enumerate(current_G.nodes())}
    expanded = expand_partition(partition, node_mapping)
    history.append(expanded)
    modularity_history.append(modularity(G, expanded))

    def on_move(node, from_comm, to_comm, delta, current_partition):
        step_counter["step"] += 1
        expanded_partition = expand_partition(current_partition, node_mapping)
        history.append(expanded_partition)
        q_total = modularity(G, expanded_partition)
        modularity_history.append(q_total)
        moves.append(
            {
                "step": step_counter["step"],
                "algorithm": "enhanced",
                "node": node,
                "from": from_comm,
                "to": to_comm,
                "delta_q": delta,
                "q_total": q_total,
            }
        )

    while True:
        partition = local_moving_queue(current_G, partition, trace_log, on_move, theta=theta)

        partition, triggered, before, after = refine_partitions(current_G, partition)
        refine_stats["triggered"] += triggered
        refine_stats["disconnected_before"] += before
        refine_stats["disconnected_after"] += after

        expanded = expand_partition(partition, node_mapping)
        history.append(expanded)
        modularity_history.append(modularity(G, expanded))

        newG = aggregate_graph(current_G, partition)
        if newG.number_of_nodes() == current_G.number_of_nodes():
            break

        hierarchy.append(
            {
                "level": len(hierarchy),
                "mapping": _community_merge_mapping(partition),
            }
        )

        new_mapping = {}
        for comm in newG.nodes():
            members = set()
            for old_node, old_comm in partition.items():
                if old_comm == comm:
                    members |= node_mapping[old_node]
            new_mapping[comm] = members

        current_G = newG
        node_mapping = new_mapping
        partition = {n: i for i, n in enumerate(current_G.nodes())}
        expanded = expand_partition(partition, node_mapping)
        history.append(expanded)
        modularity_history.append(modularity(G, expanded))

    final_partition = history[-1]
    return final_partition, history, modularity_history, trace_log, refine_stats, moves, hierarchy


def make_toy_graph() -> nx.Graph:
    G = nx.Graph()
    edges = [
        (0, 1), (0, 2), (1, 2),
        (3, 4), (3, 5), (4, 5),
        (2, 3),
    ]
    G.add_edges_from(edges, weight=1.0)
    return G


def make_fig4_toy_graph() -> nx.Graph:
    """
    Construct a toy graph to reproduce Fig.4-style issue:
    two groups (1-5 attack, 6-9 normal) connected via hub 0,
    plus another group (10-15) also connected to 0.
    """
    G = nx.Graph()
    # Attack group (1-5) as two sub-cliques that only connect via hub 0.
    attack_left = [1, 2, 3]
    attack_right = [4, 5]
    G.add_edges_from([(1, 2), (2, 3), (1, 3)], weight=1.0)
    G.add_edges_from([(4, 5)], weight=1.0)

    # Normal group (6-9) as a clique
    normal = [6, 7, 8, 9]
    for i in range(len(normal)):
        for j in range(i + 1, len(normal)):
            G.add_edge(normal[i], normal[j], weight=1.0)

    # Other activities (10-15) as a clique
    other = [10, 11, 12, 13, 14, 15]
    for i in range(len(other)):
        for j in range(i + 1, len(other)):
            G.add_edge(other[i], other[j], weight=1.0)

    # Connect all groups to hub 0
    for node in attack_left + attack_right + normal + other:
        G.add_edge(0, node, weight=1.0)

    return G


def find_first_disconnected_partition(G: nx.Graph, history: list):
    for index, partition in enumerate(history):
        comm_nodes = defaultdict(list)
        for node, comm in partition.items():
            comm_nodes[comm].append(node)
        for comm, nodes in comm_nodes.items():
            subgraph = G.subgraph(nodes)
            if subgraph.number_of_nodes() > 1 and not is_connected_subgraph(G, nodes):
                return index, comm_nodes
    return None, None


def report_disconnected_communities(G: nx.Graph, partition: dict) -> list:
    disconnected = []
    comm_nodes = defaultdict(list)
    for node, comm in partition.items():
        comm_nodes[comm].append(node)
    for comm, nodes in comm_nodes.items():
        subgraph = G.subgraph(nodes)
        if subgraph.number_of_nodes() > 1 and not is_connected_subgraph(G, nodes):
            disconnected.append((comm, nodes))
    return disconnected


def draw_partition(G, pos, partition, title, path):
    plt.figure(figsize=(5, 4))
    comms = list(sorted(set(partition.values())))
    color_map = {c: i for i, c in enumerate(comms)}
    colors = [color_map[partition[n]] for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.cm.Set3, edge_color="#999")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def render_partition(ax, G, pos, partition, title):
    comms = list(sorted(set(partition.values())))
    color_map = {c: i for i, c in enumerate(comms)}
    colors = [color_map[partition[n]] for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.cm.Set3, edge_color="#999", ax=ax)
    ax.set_title(title)


def generate_animation(G, pos, history, out_path, prefix):
    frames = []
    for i, part in enumerate(history):
        frame_path = os.path.join(os.path.dirname(out_path), f"{prefix}_frame_{i:03d}.png")
        draw_partition(G, pos, part, f"Step {i}", frame_path)
        frames.append(frame_path)

    if IMAGEIO_AVAILABLE:
        images = [imageio.imread(f) for f in frames]
        imageio.mimsave(out_path, images, fps=2)
        return out_path

    try:
        import matplotlib.animation as animation
        fig, ax = plt.subplots(figsize=(5, 4))

        def update(i):
            ax.clear()
            part = history[i]
            comms = list(sorted(set(part.values())))
            color_map = {c: j for j, c in enumerate(comms)}
            colors = [color_map[part[n]] for n in G.nodes()]
            nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.cm.Set3, edge_color="#999", ax=ax)
            ax.set_title(f"Step {i}")

        ani = animation.FuncAnimation(fig, update, frames=len(history), interval=500)
        ani.save(out_path)
        plt.close(fig)
        return out_path
    except Exception:
        return None


def generate_side_by_side_animation(G, pos, std_history, enh_history, out_path):
    frame_count = max(len(std_history), len(enh_history))
    frames = []
    out_dir = os.path.dirname(out_path)
    for i in range(frame_count):
        std_part = std_history[min(i, len(std_history) - 1)]
        enh_part = enh_history[min(i, len(enh_history) - 1)]
        frame_path = os.path.join(out_dir, f"compare_frame_{i:03d}.png")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        render_partition(axes[0], G, pos, std_part, f"Standard Step {i}")
        render_partition(axes[1], G, pos, enh_part, f"Enhanced Step {i}")
        plt.tight_layout()
        plt.savefig(frame_path)
        plt.close(fig)
        frames.append(frame_path)

    if IMAGEIO_AVAILABLE:
        images = [imageio.imread(f) for f in frames]
        imageio.mimsave(out_path, images, fps=2)
        return out_path
    return None


def plot_modularity_comparison(std_history, enh_history, out_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(std_history)), std_history, marker="o", label="Standard")
    plt.plot(range(len(enh_history)), enh_history, marker="o", label="Enhanced")
    plt.xlabel("Step")
    plt.ylabel("Modularity Q")
    plt.title("Modularity Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "modularity.png")
    plt.savefig(path)
    plt.close()
    return path


def is_connected_subgraph(G: nx.Graph, nodes_in_comm: list) -> bool:
    if not nodes_in_comm:
        return True
    return nx.is_connected(G.subgraph(nodes_in_comm))


def _community_merge_mapping(partition: dict) -> dict:
    mapping = defaultdict(list)
    for node, comm in partition.items():
        mapping[comm].append(node)
    return {str(comm): sorted(nodes) for comm, nodes in mapping.items()}


def write_moves_csv(moves, path):
    fieldnames = ["step", "algorithm", "node", "from", "to", "delta_q", "q_total"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in moves:
            writer.writerow(row)


def main():
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    seed = 42
    theta = 0.5
    G = make_fig4_toy_graph()

    std_partition, std_history, std_modularity, std_trace, std_moves, std_hierarchy = (
        louvain_with_trace(G, seed=seed)
    )
    enh_partition, enh_history, enh_modularity, enh_trace, refine_stats, enh_moves, enh_hierarchy = (
        enhanced_louvain_with_trace(G, seed=seed, theta=theta)
    )

    std_log_path = os.path.join(out_dir, "trace_standard.log")
    with open(std_log_path, "w", encoding="utf-8") as f:
        for line in std_trace:
            f.write(line + "\n")

    enh_log_path = os.path.join(out_dir, "trace_enhanced.log")
    with open(enh_log_path, "w", encoding="utf-8") as f:
        for line in enh_trace:
            f.write(line + "\n")

    pos = nx.spring_layout(G, seed=seed)
    std_anim = generate_side_by_side_animation(
        G, pos, std_history, enh_history, os.path.join(out_dir, "louvain.gif")
    )
    enh_anim = generate_side_by_side_animation(
        G, pos, std_history, enh_history, os.path.join(out_dir, "enhanced.gif")
    )

    compare_curve = plot_modularity_comparison(std_modularity, enh_modularity, out_dir)

    break_index, _ = find_first_disconnected_partition(G, std_history)
    if break_index is not None:
        partition = std_history[break_index]
        frame_path = os.path.join(out_dir, f"broken_partition_{break_index:03d}.png")
        draw_partition(G, pos, partition, f"Disconnected at step {break_index}", frame_path)

        disconnected = report_disconnected_communities(G, partition)
        print("Disconnected communities detected (standard):")
        for comm, nodes in disconnected:
            print(f"- community {comm}: nodes={sorted(nodes)}")
    else:
        print("No disconnected communities detected in standard history.")

    print("Refine stats (enhanced):")
    print(f"- refine_triggered: {refine_stats['triggered']}")
    print(
        "- disconnected communities before/after: "
        f"{refine_stats['disconnected_before']} -> {refine_stats['disconnected_after']}"
    )

    moves_path = os.path.join(out_dir, "moves.csv")
    write_moves_csv(std_moves + enh_moves, moves_path)

    hierarchy_path = os.path.join(out_dir, "hierarchy.json")
    with open(hierarchy_path, "w", encoding="utf-8") as f:
        json.dump(
            {"standard": std_hierarchy, "enhanced": enh_hierarchy},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Done.")
    print(f"Standard final partition: {std_partition}")
    print(f"Enhanced final partition: {enh_partition}")
    print(f"Standard animation: {std_anim}")
    print(f"Enhanced animation: {enh_anim}")
    print(f"Comparison modularity curve: {compare_curve}")
    print(f"Standard trace log: {std_log_path}")
    print(f"Enhanced trace log: {enh_log_path}")
    print(f"Moves log: {moves_path}")
    print(f"Hierarchy mapping: {hierarchy_path}")


if __name__ == "__main__":
    main()
