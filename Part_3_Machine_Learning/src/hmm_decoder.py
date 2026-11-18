"""
PathSnap HMM & Viterbi Route Reconstruction Module
Implements Hidden Markov Model decoding using topology-aware transition probabilities
and model-based emission probabilities to reconstruct continuous vehicle routes.
"""

import math
import numpy as np


class RoadNetworkGraph:
    """
    Adjacency graph of the road network for fast topological transition lookups.
    """
    def __init__(self, arcs, num_nodes):
        self.arcs = arcs
        self.num_arcs = len(arcs)
        self.num_nodes = num_nodes

        # Mapping: node_id -> list of outgoing arc_ids
        self.outgoing = [[] for _ in range(num_nodes)]
        # Mapping: node_id -> list of incoming arc_ids
        self.incoming = [[] for _ in range(num_nodes)]

        for arc_id, (u, v) in enumerate(arcs):
            if 0 <= u < num_nodes:
                self.outgoing[u].append(arc_id)
            if 0 <= v < num_nodes:
                self.incoming[v].append(arc_id)

    def is_direct_connected(self, arc_prev, arc_curr):
        """Returns True if arc_prev connects directly to arc_curr."""
        if arc_prev == arc_curr:
            return True
        return self.arcs[arc_prev, 1] == self.arcs[arc_curr, 0]

    def get_topological_hop_distance(self, arc_prev, arc_curr, max_hops=3):
        """
        Calculates topological hop distance (0 for same, 1 for direct, 2 for 2-hop, 3 for 3-hop).
        Returns max_hops + 1 if unreachable within max_hops.
        """
        if arc_prev == arc_curr:
            return 0
        to_node = self.arcs[arc_prev, 1]
        target_from = self.arcs[arc_curr, 0]

        if to_node == target_from:
            return 1

        if max_hops >= 2:
            # 2-hop search
            for next_arc in self.outgoing[to_node]:
                next_to = self.arcs[next_arc, 1]
                if next_to == target_from:
                    return 2

        if max_hops >= 3:
            # 3-hop search
            for next_arc in self.outgoing[to_node]:
                for hop2_arc in self.outgoing[self.arcs[next_arc, 1]]:
                    if self.arcs[hop2_arc, 1] == target_from:
                        return 3

        return max_hops + 1


class HMMRouteDecoder:
    """
    Hidden Markov Model Map-Matching Decoder using Viterbi algorithm.
    """
    def __init__(
        self,
        network_graph,
        sigma_gps=20.0,
        beta_transition=10.0,
        model_weight=0.6,
        disconnected_penalty=-15.0,
        max_time_gap=10.0
    ):
        self.graph = network_graph
        self.sigma_gps = sigma_gps
        self.beta = beta_transition
        self.alpha = model_weight
        self.disconnected_penalty = disconnected_penalty
        self.max_time_gap = max_time_gap

    def compute_log_emission(self, cand, model_prob=None):
        """
        Computes fused log-emission probability combining spatial Gaussian noise
        and classifier predicted probability.
        """
        dist = cand["dist"]
        # Spatial Gaussian log-likelihood
        log_spatial = -0.5 * (dist / self.sigma_gps) ** 2

        if model_prob is not None:
            eps = 1e-6
            log_model = math.log(max(eps, min(1.0 - eps, model_prob)))
            log_emission = self.alpha * log_model + (1.0 - self.alpha) * log_spatial
        else:
            log_emission = log_spatial

        return log_emission

    def compute_log_transition(self, cand_prev, cand_curr, dt, d_euc):
        """
        Computes log-transition probability based on network topological connectivity
        and physical distance consistency.
        """
        # If large GPS gap, uncouple transitions (soft reset)
        if dt > self.max_time_gap or d_euc > 300.0:
            return 0.0

        arc_prev = cand_prev["arc_id"]
        arc_curr = cand_curr["arc_id"]

        # Case 1: Identical road segment
        if arc_prev == arc_curr:
            # Expected distance is d_euc; difference is ~0
            return 0.0

        # Case 2: Direct 1-hop connection
        if self.graph.is_direct_connected(arc_prev, arc_curr):
            # Normal road progress
            return -0.5

        # Case 3: Multi-hop reachability
        hops = self.graph.get_topological_hop_distance(arc_prev, arc_curr, max_hops=3)
        if hops == 2:
            return -2.5
        elif hops == 3:
            return -5.0

        # Case 4: Disconnected or wrong direction
        return self.disconnected_penalty

    def decode(self, trip_candidates_list, track_kinematics, model_probs_list=None):
        """
        Performs Viterbi decoding over the candidate trellis for the trip.
        Returns:
            best_path: list of decoded arc_ids of length T
        """
        T = len(trip_candidates_list)
        if T == 0:
            return []

        # Initialize trellis Viterbi log-probabilities and backpointers
        # V[t] = dict mapping candidate_idx -> max_log_prob
        # bp[t] = dict mapping candidate_idx -> prev_candidate_idx

        V = []
        bp = []

        # Step t = 0
        cands_0 = trip_candidates_list[0]
        if not cands_0:
            return [-1] * T

        probs_0 = model_probs_list[0] if model_probs_list is not None else [None] * len(cands_0)
        v0 = {}
        for j, c in enumerate(cands_0):
            p = probs_0[j] if j < len(probs_0) else None
            v0[j] = self.compute_log_emission(c, p)

        V.append(v0)
        bp.append({})

        # Steps t = 1 ... T - 1
        for t in range(1, T):
            cands_t = trip_candidates_list[t]
            if not cands_t:
                # Fallback: copy previous
                V.append(V[-1])
                bp.append({0: 0})
                continue

            probs_t = model_probs_list[t] if model_probs_list is not None else [None] * len(cands_t)
            dt = track_kinematics["time_gaps"][t]
            d_euc = track_kinematics["distances_m"][t]

            v_prev = V[t - 1]
            cands_prev = trip_candidates_list[t - 1]

            vt = {}
            bpt = {}

            for j, curr_c in enumerate(cands_t):
                prob_j = probs_t[j] if j < len(probs_t) else None
                log_emis = self.compute_log_emission(curr_c, prob_j)

                best_val = -1e12
                best_i = 0

                for i, prev_c in enumerate(cands_prev):
                    prev_score = v_prev.get(i, -1e12)
                    log_trans = self.compute_log_transition(prev_c, curr_c, dt, d_euc)
                    score = prev_score + log_trans + log_emis

                    if score > best_val:
                        best_val = score
                        best_i = i

                vt[j] = best_val
                bpt[j] = best_i

            V.append(vt)
            bp.append(bpt)

        # Backtrack best path
        best_last_idx = max(V[-1].keys(), key=lambda k: V[-1][k])
        decoded_path = [trip_candidates_list[-1][best_last_idx]["arc_id"]]

        curr_idx = best_last_idx
        for t in range(T - 1, 0, -1):
            curr_idx = bp[t].get(curr_idx, 0)
            cands = trip_candidates_list[t - 1]
            arc_id = cands[curr_idx]["arc_id"] if curr_idx < len(cands) else -1
            decoded_path.append(arc_id)

        decoded_path.reverse()
        return decoded_path
