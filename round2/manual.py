from typing import Tuple, List
from math import log

rates = [
    [1, 0.48, 1.52, 0.71],
    [2.05, 1, 3.26, 1.56],
    [0.64, 0.3, 1, 0.46],
    [1.41, 0.61, 2.08, 1]
]

currencies = ('Pizza', 'Wasabi', 'Snowball', 'Shells')


def negate_logarithm_convertor(graph: Tuple[Tuple[float]]) -> List[List[float]]:
    result = [[-log(edge) for edge in row] for row in graph]
    return result


def arbitrage(currency_tuple: tuple, rates_matrix: Tuple[Tuple[float, ...]]):

    trans_graph = negate_logarithm_convertor(rates_matrix)

    source = 0
    n = len(trans_graph)
    min_dist = [float('inf')] * n

    pre = [-1] * n

    min_dist[source] = source

    # 'Relax edges |V-1| times'
    for _ in range(n - 1):
        for source_curr in range(n):
            for dest_curr in range(n):
                if min_dist[dest_curr] > min_dist[source_curr] + trans_graph[source_curr][dest_curr]:
                    min_dist[dest_curr] = min_dist[source_curr] + trans_graph[source_curr][dest_curr]
                    pre[dest_curr] = source_curr

    # if we can still relax edges, then we have a negative cycle
    for source_curr in range(n):
        for dest_curr in range(n):
            if min_dist[dest_curr] > min_dist[source_curr] + trans_graph[source_curr][dest_curr]:
                # negative cycle exists, and use the predecessor chain to print the cycle
                print_cycle = [dest_curr, source_curr]
                # Start from the source and go backwards until you see the source vertex again or any vertex that already exists in print_cycle array
                while pre[source_curr] not in print_cycle:
                    print_cycle.append(pre[source_curr])
                    source_curr = pre[source_curr]
                print_cycle.append(pre[source_curr])
                print("Arbitrage Opportunity: \n")
                print(" --> ".join([currencies[p] for p in print_cycle[::-1]]))


if __name__ == "__main__":
    arbitrage(currencies, rates)
