"""
This script computes similarity between a proof of concept (PoC) IR and a series of mutated variant 
IRa by node/block sequence alignment. For a graph type at a optimization phase, the PoC's graph and the mutant's
graph are each turned into a sequence of node semantic signatures ordered by node
id in ascending order, then aligned with the Needleman-Wunsch global alignment
algorithm.

The implementation works for SoN graphs and CFGs. This means that any Turbofan, Turboshaft, or maglev
IRs will work based apon what graph type is recorded from extract_ir.py

    "graph"             TurboFan sea-of-nodes. Split by edge type into the
                        value / control / effect / context / frame-state graphs,
                        scored independently (after_<gtype>.json).
    "turboshaft_graph"  Turboshaft CFG. Operations are already scheduled into basic
                        blocks and edges carry no type, so there is nothing to split:
                        one "cfg" graph per phase (after_cfg.json).

Both use the same pipeline -- id-ordered sequence of semantic signatures, then
Needleman-Wunsch as a minimum edit distance (match 0, substitution 1, indel 1), then
similarity = 1 - distance / max(len1, len2). Only the node signature differs, because
a sea-of-nodes node and a Turboshaft operation share no attributes (see
IDENT_FIELDS vs CFG_IDENT_FIELDS, and cfg_sem_sig for how block membership stands in
for the sea-of-nodes "control" field).

Usage:
    python similarity.py extracted/
    python similarity.py extracted/ --reference turbo-buggy
    python similarity.py --bug <bug dir name>
"""

import json
import os
import re
import sys
import csv
import argparse
from collections import defaultdict

from paths import bug_paths

try:
    from scipy.optimize import linear_sum_assignment as _scipy_assignment
except ImportError:
    _scipy_assignment = None


IDENT_FIELDS = ("opcode", "type", "control", "live", "properties")

CFG_IDENT_FIELDS = ("title", "type", "op_effects", "representation", "properties")

# Instruction identity: the operation's title (opcode), properties, and machine
# representation -- purely the op's OWN content, with no data-flow neighbors. An
# operation's token no longer depends on what it is wired to, so two ops of the
# same kind match even when their operands differ. (`type` is dropped: it is null
# on essentially every Turboshaft op. Neighbors are dropped: they made the token
# over-strict -- see cfg_block_instruction_sequences.)
INSTR_IDENT_FIELDS = ("title", "properties", "representation")

SON_KIND = "graph"
CFG_KIND = "turboshaft_graph"

# Blend weight for the CFG cell score (cfg_block_mean_similarity):
#     similarity = CFG_ALPHA * sim_struct + (1 - CFG_ALPHA) * instr_term
# CFG_ALPHA weights block-structure change against in-block instruction change.
# One global value -- do NOT tune per-bug (only one CFG ground-truth bug exists, so
# a per-bug weight overfits). Start at 0.5 and adjust from the alpha sweep.
CFG_ALPHA = 0.5


def graph_kind(graph):
    """
    Decide whether one extracted graph is sea-of-nodes or a Turboshaft CFG.

    extract_ir.py stamps GRAPH_KIND on CFG graphs, which is the authoritative answer.
    Falls back to the presence of BLOCKS, since only Turboshaft phases have them.
    """
    kind = graph.get("GRAPH_KIND")
    if kind in (SON_KIND, CFG_KIND):
        return kind
    return CFG_KIND if "BLOCKS" in graph else SON_KIND


# V8 prints raw heap and external addresses straight into node text:
#
#     type  = "HeapConstant(0x59fe08fcaba8)"
#     title = "HeapConstant[0x59fe08fd6da0 <JSFunction foo (sfi = 0x59fe08fc21e8)>]"
#
# Those addresses are volatile -- memory allocation order changes every
# run -- so the same node dumped from two runs gets two different idents and scores
# as a mismatch. Measured on typer_hardening_bypass, three fields carry them
# (`type`, `title`, `label`) across HeapConstant, CompressedHeapConstant,
# FrameState, ExternalConstant, JSLoadGlobal and JSStoreGlobal.
_VOLATILE_HEX_RE = re.compile(r'0x[0-9a-fA-F]+')

# What a stripped address becomes. Any fixed string works; this one is visibly a
# placeholder in the alignment dumps rather than looking like a real address.
VOLATILE_HEX_PLACEHOLDER = "0x?"

# V8 also prints the compiled function's OWN NAME into node text, inside its object
# wrappers, after the (already-blanked) heap address:
#
#     properties = "[heap object: 0x? <JSFunction asciiArt (sfi = 0x?)>]"        (Constant)
#     properties = "[..., 0x? <SharedFunctionInfo asciiArt>, state values: ...]" (FrameState)
#
# The name is chosen by the PoC, so a mutation that ONLY renames the function -- a pure
# rename with no behavior change -- otherwise scores every JSFunction / SharedFunctionInfo
# op as a mismatch and drops the rename-invariant similarity below 1.0. Blanking the
# identifier right after these wrappers restores it. This is a no-op whenever both graphs
# carry the same name (every non-rename mutant keeps the seed's function name on BOTH
# sides), so it only ever collapses a pure-rename difference -- it never merges two
# genuinely different functions in the same comparison beyond their names. The name token
# is [^\s()>]+: it stops at the space before "(sfi ...)" and at the closing ">".
_VOLATILE_NAME_RE = re.compile(
    r'(<(?:JSFunction|SharedFunctionInfo|JSBoundFunction)\s+)[^\s()>]+'
)
VOLATILE_NAME_PLACEHOLDER = "?"


def normalize_volatile(value):
    """
    Blank out run-to-run-volatile text in one field value, leaving the semantic text
    around it intact:

        "HeapConstant[0x59fe08fd6da0 <JSFunction foo (sfi = 0x59fe08fc21e8)>]"
     -> "HeapConstant[0x? <JSFunction ? (sfi = 0x?)>]"

    Two normalizations, in order:
      1. Heap addresses -> "0x?".  Every 0x value in this IR is a POINTER (numeric
         constants print in decimal), so no literal value is erased. Bytecode offsets
         and slot counts ("FrameState[INTERPRETED_FRAME, 28, ...]", "<NativeContext[248]>")
         are decimal and survive.
      2. The function name inside <JSFunction NAME ...> / <SharedFunctionInfo NAME> ->
         "?", so a pure function rename (asciiArt -> mutantArt) no longer mismatches.
         See _VOLATILE_NAME_RE for why this is a no-op for non-rename mutants.

    Non-strings pass through. This is the single chokepoint for every field set
    (IDENT_FIELDS, CFG_IDENT_FIELDS, INSTR_IDENT_FIELDS), so both the SON and the CFG
    alignments -- score and print_alignment dump alike -- inherit both rules.
    """
    if isinstance(value, str):
        value = _VOLATILE_HEX_RE.sub(VOLATILE_HEX_PLACEHOLDER, value)
        value = _VOLATILE_NAME_RE.sub(r'\1' + VOLATILE_NAME_PLACEHOLDER, value)
        return value
    return value


def node_ident(node, fields=IDENT_FIELDS):
    """
    The identity of one IR node: the tuple of fields that must match for two
    nodes to be considered "the same kind of node".

        SON: (opcode, type, control, live, properties)
        CFG: (title, type, op_effects, representation, properties)

    This is what both a node's own key and each of its neihbors' keys are built
    from, so a change to any of these fields on a node or on one of its neighbors
    shows up as a mismatch.

    Every field is passed through normalize_volatile first, so run-to-run heap
    addresses never reach a comparison. This is the single chokepoint for all three
    field sets (IDENT_FIELDS, CFG_IDENT_FIELDS, INSTR_IDENT_FIELDS) and it also
    covers neighbor idents, which are built from this same function.
    """
    return tuple(normalize_volatile(node.get(f)) for f in fields)


def son_sem_sig(node, neighbor_ids):
    """
    Sea-of-nodes semantic signature of one node, as a hashable tuple:

        (ident, degree)

    - ident  : node_ident(node) = (opcode, type, control, live, properties) -- what
               kind of node it is
    - degree : how many edges touch it -- incoming AND outgoing

    Deliberately NOT neighbor-aware (unlike node_sem_sig below): a node's token does
    not fold in its neighbors' idents. A change to one node therefore no longer
    ripples into every neighbor's token, which used to inflate the edit distance by
    double-counting. degree still captures a change in edge COUNT; a constant-degree
    rewiring (an edge repointed between two same-ident nodes) is intentionally
    invisible -- the accepted tradeoff for less double-counting.

    Two nodes with equal signatures are treated as a match by Needleman-Wunsch.
    """
    return (node_ident(node), len(neighbor_ids))


def node_sem_sig(node, ident_by_id, neighbor_ids, fields=IDENT_FIELDS):
    """
    Semantic signature of one IR node, as a hashable tuple:

        (ident, degree, (neighbor_ident, neighbor_ident, ...))

    - ident          : node_ident(node) = (opcode, type, control, live,
                       properties) -- what kind of node it is
    - degree         : how many edges touch it -- incoming AND outgoing
    - neighbor_idents: the ident of the node on the other end of each incident
                       edge, in edge order (neighbors are identified by the same
                       ident tuple -- we do NOT recurse into the neighbor's own
                       edges)

    parameters:
        node        : the node dict
        ident_by_id : {node_id: node_ident(...)} for the whole graph, to resolve
                      neighbors
        neighbor_ids: list of this node's neighbor ids (one entry per incident
                      edge, in/out combined), in edge order

    Two nodes with equal signatures are treated as a match by Needleman-Wunsch.
    """
    ident = node_ident(node, fields)
    neighbor_idents = tuple(ident_by_id.get(d) for d in neighbor_ids)
    return (ident, len(neighbor_ids), neighbor_idents)


def cfg_sem_sig(op, ident_by_id, neighbor_ids, block_sig):
    """
    Semantic signature of one Turboshaft operation:

        (ident, degree, (neighbor_ident, ...), block_sig)

    Identical in shape to the sea-of-nodes signature, with one field appended.

    """
    ident = node_ident(op, CFG_IDENT_FIELDS)
    neighbor_idents = tuple(ident_by_id.get(d) for d in neighbor_ids)
    return (ident, len(neighbor_ids), neighbor_idents, block_sig)


def _incident_neighbors(graph, id_to_node):
    """
    {node_id: [neighbor_id, ...]} -- one entry per incident edge, in edge order.

    Every incident edge contributes a neighbor to BOTH endpoints, so degree counts
    incoming and outgoing alike. Keeps only edges whose endpoints are both real nodes
    in this graph, so every neighbor has a known ident.
    """
    neighbors = defaultdict(list)
    for e in graph.get("EDGES", []):
        u, v = e.get("from"), e.get("to")
        if u in id_to_node and v in id_to_node:
            neighbors[u].append(v)   # v is a neighbor of u (outgoing)
            neighbors[v].append(u)   # u is a neighbor of v (incoming)
    return neighbors


def _real_nodes(graph):
    nodes = [n for n in graph.get("NODES", [])
             if isinstance(n, dict) and n.get("id") is not None]
    return {n["id"]: n for n in nodes}


def build_sequence_son(graph):
    """Sea-of-nodes: sequence of {node_id: sem_sig}, ordered by node id ascending."""
    id_to_node = _real_nodes(graph)
    neighbors = _incident_neighbors(graph, id_to_node)

    return [{nid: son_sem_sig(id_to_node[nid], neighbors[nid])}
            for nid in sorted(id_to_node)]


def build_sequence_cfg(graph):
    """
    Turboshaft CFG: sequence of {op_id: cfg_sem_sig}, ordered by operation id ascending.

    Same ordering rule as the SON path. Note this is a stronger guarantee here than it
    is there: Turboshaft is scheduled, so ascending operation id is the program order
    within a block rather than an arbitrary tie-break over an unordered graph.
    """
    id_to_op = _real_nodes(graph)
    ident_by_id = {nid: node_ident(op, CFG_IDENT_FIELDS) for nid, op in id_to_op.items()}
    neighbors = _incident_neighbors(graph, id_to_op)

    # An operation's control context = the kind and in-degree of its block.
    block_sig_by_id = {
        b.get("id"): (b.get("kind"), len(b.get("predecessors") or []))
        for b in graph.get("BLOCKS", []) or []
        if isinstance(b, dict)
    }

    return [
        {nid: cfg_sem_sig(
            id_to_op[nid],
            ident_by_id,
            neighbors[nid],
            block_sig_by_id.get(id_to_op[nid].get("block_id")),
        )}
        for nid in sorted(id_to_op)
    ]


def build_sequence(graph):
    """
    Turn one graph into the sequence Needleman-Wunsch aligns: a list of
    single-entry {node_id: sem_sig} dicts, ordered by node id ascending.

        e.g. [{1: sig1}, {4: sig4}, {5: sig5}]

    Dispatches on the graph's kind, which extract_ir.py records from the raw dump's
    phase["type"]: sea-of-nodes ("graph") and Turboshaft ("turboshaft_graph") are
    different representations and cannot share a node identity.

    parameters:
        graph: SON -- {"NODES": [...], "EDGES": [{"from","to","kind"}...]}
               CFG -- {"NODES": [...], "EDGES": [...], "BLOCKS": [...],
                       "GRAPH_KIND": "turboshaft_graph"}
               as produced by extract_ir.py (after_<gtype>.json / after_cfg.json).
    """
    if graph_kind(graph) == CFG_KIND:
        return build_sequence_cfg(graph)
    return build_sequence_son(graph)


def _sem(element):
    """Extract the sem_sig from a {node_id: sem_sig} sequence element."""
    return next(iter(element.values()))


# Edit costs. A match is free; a substitution (two aligned nodes with different
# sem_sigs), a deletion (a seed node with no mutant partner), and an insertion
# (a mutant node with no seed partner) each cost one edit.
SUB_COST = 1
GAP_COST = 1
def needleman_wunsch(seq1, seq2, sub=SUB_COST, gap=GAP_COST):
    """
    Needleman-Wunsch global alignment over two sequences of {node_id: sem_sig}
    elements, run as a minimum edit distance.
    Two elements match (cost 0) iff their sem_sigs are equal; otherwise aligning
    them is a substitution (cost `sub`). A gap on either side is an indel (cost
    `gap`). Returns the two aligned sequences and the optimal edit distance.
    """
    n = len(seq1)
    m = len(seq2)

    # D[i][j] = min edits to turn seq1[:i] into seq2[:j].
    score_matrix = []
    pointer_matrix = []
    for i in range(n+1):
        temp = []
        for j in range(m+1):
            temp.append(0)
        score_matrix.append(temp)

    for i in range(n+1):
        temp = []
        for j in range(m+1):
            temp.append("")
        pointer_matrix.append(temp)

    # Borders: turning i nodes into nothing is i deletions; nothing into j is j
    # insertions. (These are NOT zero -- that is the key difference from a
    # reward-maximizing NW with a free gap.)
    for i in range(1, n + 1):
        score_matrix[i][0] = i * gap
        pointer_matrix[i][0] = "UP"

    for j in range(1, m + 1):
        score_matrix[0][j] = j * gap
        pointer_matrix[0][j] = "LEFT"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if _sem(seq1[i - 1]) == _sem(seq2[j - 1]):
                diag = score_matrix[i - 1][j - 1] + 0        # match, free
            else:
                diag = score_matrix[i - 1][j - 1] + sub      # substitution

            up = score_matrix[i - 1][j] + gap                # delete seq1[i-1]
            left = score_matrix[i][j - 1] + gap              # insert seq2[j-1]

            min_score = min(diag, up, left)
            score_matrix[i][j] = min_score

            if min_score == diag:
                pointer_matrix[i][j] = "DIAG"
            elif min_score == up:
                pointer_matrix[i][j] = "UP"
            else:
                pointer_matrix[i][j] = "LEFT"

    align1 = []
    align2 = []
    i = n
    j = m
    while i > 0 or j > 0:
        if pointer_matrix[i][j] == "DIAG":
            align1.append(seq1[i - 1])
            align2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif pointer_matrix[i][j] == "UP":
            align1.append(seq1[i - 1])
            align2.append("-")
            i -= 1
        else:# pointer_matrix[i][j] == "LEFT":
            align1.append("-")
            align2.append(seq2[j - 1])
            j -= 1

    align1.reverse()
    align2.reverse()

    return align1, align2, score_matrix[n][m]


def align(seq1, seq2, sub=SUB_COST, gap=GAP_COST):
    """
    Align two already sorted IR sequences with Needleman-Wunsch and turn the
    optimal edit distance into a similarity in [0, 1] via

        similarity = 1 - distance / max(len(seq1), len(seq2)).

    Because the worst-case edit distance is exactly max(len1, len2), the result
    is always in [0, 1] with no clamping. Two empty sequences are defined as
    identical (1.0); one empty and one non-empty is 0.0 (all insertions).

    Returns:
        dict with aligned_ref, aligned_mut, distance, max_distance, similarity.
    """
    aligned_ref, aligned_mut, distance = needleman_wunsch(seq1, seq2, sub, gap)

    n1, n2 = len(seq1), len(seq2)
    max_distance = max(n1, n2)
    if max_distance == 0:
        similarity = 1.0                       # both empty -> identical
    else:
        similarity = 1.0 - distance / max_distance

    return {
        "aligned_ref": aligned_ref,
        "aligned_mut": aligned_mut,
        "distance": distance,
        "max_distance": max_distance,
        "similarity": similarity,
    }

def cfg_block_sequence(graph):
    """
    Step 1 input: one CFG's blocks as a Needleman-Wunsch sequence,

        [{block_id: block_sig}, ...]  ordered by block id ascending

    where block_sig = (kind, number_of_predecessors) -- the same structural
    identity build_sequence_cfg folds into each operation. Ordering by block id
    mirrors the operation-level rule (ascending id = schedule order in Turboshaft),
    so the outer alignment behaves exactly like the inner one, one level up.
    """
    blocks = [b for b in graph.get("BLOCKS", []) or []
              if isinstance(b, dict) and b.get("id") is not None]
    return [
        {b["id"]: (b.get("kind"), len(b.get("predecessors") or []))}
        for b in sorted(blocks, key=lambda b: b["id"])
    ]


def align_cfg_blocks(ref_graph, mut_graph, sub=SUB_COST, gap=GAP_COST):
    """
    Step 1: align the block sequences of the seed (ref) and mutant CFGs.

    Reuses needleman_wunsch: two blocks match (cost 0) iff their block_sigs are
    equal, otherwise the diagonal is a substitution and a gap is an indel. The
    result is a partial injection between the two block sets -- a bijection only
    in the special case where every block matches and no block is left over.

    Returns a list of (ref_block_id, mut_block_id) pairs, one per alignment slot:
        both ids present -> a matched/corresponding block pair
        one id None      -> an unmatched block (the "e" that gets left out)
    """
    a_ref, a_mut, _distance = needleman_wunsch(
        cfg_block_sequence(ref_graph), cfg_block_sequence(mut_graph), sub, gap
    )
    pairs = []
    for r, m in zip(a_ref, a_mut):
        rid = None if r == "-" else next(iter(r))
        mid = None if m == "-" else next(iter(m))
        pairs.append((rid, mid))
    return pairs


def cfg_instructions_by_block(graph):
    """
    Group one CFG's operations by their block, each block's ops as a Step-2
    sequence:

        {block_id: [{op_id: instruction_sig}, ...] ordered by op id ascending}

    instruction_sig is the operation's own content only --
    (ident, degree, neighbor_idents) via node_sem_sig over CFG_IDENT_FIELDS --
    and deliberately omits block_sig: block context is Step 1's responsibility
    now, so it must not be double-counted inside each operation. neighbor_idents
    still follow the data-flow edges, which may cross block boundaries; that is
    intended, it captures each operation's data dependencies.
    """
    id_to_op = _real_nodes(graph)
    ident_by_id = {nid: node_ident(op, CFG_IDENT_FIELDS) for nid, op in id_to_op.items()}
    neighbors = _incident_neighbors(graph, id_to_op)

    by_block = defaultdict(list)
    for nid in sorted(id_to_op):
        op = id_to_op[nid]
        sig = node_sem_sig(op, ident_by_id, neighbors[nid], fields=CFG_IDENT_FIELDS)
        by_block[op.get("block_id")].append({nid: sig})
    return by_block


def align_cfg(ref_graph, mut_graph, sub=SUB_COST, gap=GAP_COST):
    """
    Steps 1 + 2 for one CFG pair, WITHOUT rolling up a similarity score.

    Step 1 pairs the blocks; Step 2 aligns the instructions of each matched pair.
    Returns one record per Step-1 alignment slot:

        {
          "ref_block": ref_block_id | None,   # None => this slot is a mutant-only block
          "mut_block": mut_block_id | None,   # None => this slot is a seed-only block
          "instructions": None                # unmatched block: nothing aligned
                        | {                   # matched block pair:
                              "aligned_ref": [...], "aligned_mut": [...],
                              "distance": int, "max_distance": int,
                          },
        }

    The final similarity (how to combine these per-block instruction alignments,
    and how to charge for unmatched blocks) is intentionally left to the next step.
    """
    ref_instr = cfg_instructions_by_block(ref_graph)
    mut_instr = cfg_instructions_by_block(mut_graph)

    records = []
    for rid, mid in align_cfg_blocks(ref_graph, mut_graph, sub, gap):
        rec = {"ref_block": rid, "mut_block": mid, "instructions": None}
        if rid is not None and mid is not None:
            rseq = ref_instr.get(rid, [])
            mseq = mut_instr.get(mid, [])
            a_ref, a_mut, distance = needleman_wunsch(rseq, mseq, sub, gap)
            rec["instructions"] = {
                "aligned_ref": a_ref,
                "aligned_mut": a_mut,
                "distance": distance,
                "max_distance": max(len(rseq), len(mseq)),
            }
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# CFG block-level similarity via approximate graph edit distance
# (bipartite GED: block assignment solved with the Hungarian algorithm).
# ---------------------------------------------------------------------------

# Relabeling cost for substituting one block for another. Ignored (0) for now:
# this level scores topology only, so matching two blocks charges nothing beyond
# their neighborhood differences. The node-level implementation will replace
# this with a real block-content cost later.
BLOCK_RELABEL_COST = 0

# Stand-in for "forbidden" cells in the cost matrix (off-diagonal deletions and
# insertions). Large enough that no optimal assignment ever uses one, small
# enough that float arithmetic on it stays exact.
_FORBIDDEN = 10**9


def cfg_block_neighbors(graph):
    """
    Adjacency of the block graph, from BLOCKS[].predecessors (integer block ids).

    Returns (preds, succs, in_deg, out_deg, n_edges):
        preds/succs: {block_id: set of neighboring block ids}  (sets: ON / IN)
        in_deg/out_deg: {block_id: edge count, WITH multiplicity}  (|IE| / |OE|)
        n_edges: total control-flow edges (|E|), with multiplicity.
    """
    blocks = [b for b in graph.get("BLOCKS", []) or []
              if isinstance(b, dict) and b.get("id") is not None]
    ids = [b["id"] for b in blocks]

    preds = {bid: set() for bid in ids}
    succs = {bid: set() for bid in ids}
    in_deg = {bid: 0 for bid in ids}
    out_deg = {bid: 0 for bid in ids}
    n_edges = 0

    for b in blocks:
        bid = b["id"]
        for p in b.get("predecessors") or []:
            preds[bid].add(p)
            in_deg[bid] += 1
            n_edges += 1
            if p in succs:
                succs[p].add(bid)
                out_deg[p] += 1
    return preds, succs, in_deg, out_deg, n_edges


def hungarian(cost):
    """
    Minimum-cost perfect assignment on a square matrix (Kuhn-Munkres with
    potentials, O(n^3)). Returns (total_cost, assignment) where assignment[row]
    = column matched to that row.
    """
    n = len(cost)
    if n == 0:
        return 0, []
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)      # p[col] = row matched to col (1-based; 0 = none)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            row = cost[i0 - 1]
            for j in range(1, n + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    assignment = [0] * n
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    total = sum(cost[i][assignment[i]] for i in range(n))
    return total, assignment


def cfg_block_cost_matrix(g1, g2):
    """
    The (|V1|+|V2|) x (|V1|+|V2|) edit-cost supermatrix over the two block sets.

        [ substitution |V1|x|V2| | deletion  |V1|x|V1| ]
        [ insertion    |V2|x|V2| | zeros     |V2|x|V1| ]

    substitution a_ij = BLOCK_RELABEL_COST
                        + |ON_i| + |ON_j| - 2|ON_i & ON_j|
                        + |IN_i| + |IN_j| - 2|IN_i & IN_j|
        (neighbor sets intersected by block id)
    deletion  (diagonal) a_ii = 1 + |OE_i| + |IE_i|; off-diagonal forbidden
    insertion (diagonal) a_jj = 1 + |OE_j| + |IE_j|; off-diagonal forbidden

    Returns (matrix, ids1, ids2) with ids in row/column order.
    """
    preds1, succs1, in1, out1, _ = cfg_block_neighbors(g1)
    preds2, succs2, in2, out2, _ = cfg_block_neighbors(g2)
    ids1 = sorted(preds1)
    ids2 = sorted(preds2)
    n1, n2 = len(ids1), len(ids2)
    size = n1 + n2

    C = [[_FORBIDDEN] * size for _ in range(size)]

    # Top-left: substitution.
    for a, i in enumerate(ids1):
        for b, j in enumerate(ids2):
            on = len(succs1[i]) + len(succs2[j]) - 2 * len(succs1[i] & succs2[j])
            inn = len(preds1[i]) + len(preds2[j]) - 2 * len(preds1[i] & preds2[j])
            C[a][b] = BLOCK_RELABEL_COST + on + inn

    # Top-right: deletion of V1 blocks (diagonal only).
    for a, i in enumerate(ids1):
        C[a][n2 + a] = 1 + out1[i] + in1[i]

    # Bottom-left: insertion of V2 blocks (diagonal only).
    for b, j in enumerate(ids2):
        C[n1 + b][b] = 1 + out2[j] + in2[j]

    # Bottom-right: dummy-to-dummy, free.
    for a in range(n2):
        for b in range(n1):
            C[n1 + a][n2 + b] = 0

    return C, ids1, ids2


def _block_type_map(graph):
    """
    {block_id: kind} from BLOCKS[] (e.g. 'MERGE' / 'LOOP' / 'BLOCK'). The block's
    type is stored under the "kind" key (the "type" field is unrelated and null).
    Used only to tie-break the block assignment toward same-kind pairings; never
    enters the base edit cost.
    """
    return {b["id"]: b.get("kind")
            for b in graph.get("BLOCKS", []) or []
            if isinstance(b, dict) and b.get("id") is not None}


def cfg_block_ged_similarity(g1, g2):
    """
    Block-level CFG similarity from the Hungarian assignment over the edit-cost
    supermatrix:

        similarity = 1 - distance / (|V1| + 2|E1| + |V2| + 2|E2|)

    The edge term is DOUBLED because a block deletion costs 1 + out_deg + in_deg, so
    summed over blocks every edge is charged to both endpoints (Sum(out+in) = 2|E|).
    The denominator is therefore the true maximum edit distance (delete-all g1 +
    insert-all g2), which the assignment can never exceed (a substitution is always
    strictly cheaper than delete+insert). So similarity is genuinely in [0, 1] --
    unlike the old |V1|+|E1|+|V2|+|E2| denominator, which under-counted edges and
    went negative for heavily-restructured graphs.

    Returns {"distance", "denominator", "similarity", "matched", "deleted",
    "inserted"} where matched is [(id1, id2), ...] and deleted/inserted list the
    block ids the assignment chose to remove from g1 / add for g2.
    """
    _, _, _, _, e1 = cfg_block_neighbors(g1)
    _, _, _, _, e2 = cfg_block_neighbors(g2)
    C, ids1, ids2 = cfg_block_cost_matrix(g1, g2)
    n1, n2 = len(ids1), len(ids2)

    # LEXICOGRAPHIC TIE-BREAK: base cost  >>  same-type  >>  same-position. Both tiers
    # are tiny penalties added ONLY to steer `assignment` among otherwise-equal choices;
    # the reported `distance`/`similarity` is recomputed from the base matrix C below.
    # The magnitudes are nested so each tier can only separate ties the tier above left:
    #   eps_t  total <= m*eps_t < 1        -> can't beat an integer base-cost gap (>=1)
    #   eps_p  total <= m*eps_p < eps_t     -> can't beat a single same-type decision
    # so type dominates position, and both are powerless against a real cost difference.
    #   * type     : penalty eps_t when the two blocks' `kind` differs (MERGE/LOOP/...).
    #   * position : penalty eps_p * |normalized rank distance|. ids1/ids2 are sorted
    #                ascending, so the enumerate index IS the block's schedule position;
    #                normalized to [0,1], the ENTRY (rank 0) matches the ENTRY and a
    #                unique exit matches the exit -- the anchors that are actually
    #                reliable. (Mid-graph position shifts when blocks are added/removed,
    #                which is why position is a last-resort tie-break, never primary.)
    Ceff = C
    if n1 and n2:
        ty1, ty2 = _block_type_map(g1), _block_type_map(g2)
        m = min(n1, n2)
        eps_t = 1.0 / (m + 1)                 # type tier
        eps_p = eps_t / (m + 1)               # position tier (strictly below type)
        def _npos(idx, n):                    # normalized schedule rank in [0,1]
            return idx / (n - 1) if n > 1 else 0.0
        Ceff = [row[:] for row in C]
        for a, i in enumerate(ids1):
            for b, j in enumerate(ids2):
                pen = eps_t if ty1.get(i) != ty2.get(j) else 0.0
                pen += eps_p * abs(_npos(a, n1) - _npos(b, n2))
                Ceff[a][b] += pen

    if not C:  # both graphs empty: nothing to assign (scipy rejects a 0x0 input)
        assignment = []
    elif _scipy_assignment is not None:
        _rows, cols = _scipy_assignment(Ceff)
        assignment = [int(c) for c in cols]
    else:
        _dist, assignment = hungarian(Ceff)

    # Base-cost distance on the chosen assignment: the tie-break changed only WHICH
    # blocks match, not the edit cost (sum over an empty C is 0 for two empty graphs).
    distance = sum(C[i][assignment[i]] for i in range(len(C)))

    matched, deleted, inserted = [], [], []
    for row, col in enumerate(assignment):
        if row < n1 and col < n2:
            matched.append((ids1[row], ids2[col]))
        elif row < n1:
            deleted.append(ids1[row])
        elif col < n2:
            inserted.append(ids2[col])

    # Edge term doubled: the numerator charges each edge to both endpoints (a block
    # deletion costs 1+out+in), so the max distance is |V1|+2|E1|+|V2|+2|E2|. Using
    # that as the denominator keeps similarity in [0,1] (no more negative-then-clamped).
    denominator = n1 + 2 * e1 + n2 + 2 * e2
    similarity = 1.0 if denominator == 0 else 1 - distance / denominator
    return {
        "distance": distance,
        "denominator": denominator,
        "similarity": similarity,
        "matched": matched,
        "deleted": deleted,
        "inserted": inserted,
    }


def cfg_block_instruction_sequences(graph):
    """
    Per block, the block's operations as a Needleman-Wunsch sequence:

        {block_id: [{op_id: (ident, ())}, ...] ordered by op id ascending}

    The instruction signature is (ident, ()):
      - ident : node_ident(op, INSTR_IDENT_FIELDS) = (title, properties,
                representation) -- purely the operation's OWN content.

    Data-flow NEIGHBORS are no longer part of the signature. They made the token
    over-strict: two ops of the same kind (e.g. two Returns) mismatched whenever any
    operand differed, so a genuinely similar block could score 0. The token now
    depends only on the op itself, and NW still charges for order/indels. The empty
    second slot keeps the (ident, neighbor_idents) shape that print_alignment.py
    unpacks -- it now always renders degree=0.

    Turboshaft is scheduled, so ascending op id IS program order within a block, and
    NW charges for an instruction appearing in the wrong place on top of the signature
    comparison.

    Contrast cfg_instructions_by_block(), which uses node_sem_sig over
    CFG_IDENT_FIELDS. That one belongs to the older cfg_ged_nw_similarity() rollup and
    is deliberately left alone.
    """
    id_to_op = _real_nodes(graph)
    ident_by_id = {nid: node_ident(op, INSTR_IDENT_FIELDS) for nid, op in id_to_op.items()}

    by_block = defaultdict(list)
    for nid in sorted(id_to_op):
        op = id_to_op[nid]
        by_block[op.get("block_id")].append({nid: (ident_by_id[nid], ())})
    return by_block


def cfg_block_mean_similarity(g1, g2, sub=SUB_COST, gap=GAP_COST):
    """
    Overall CFG similarity as a tunable blend of a structural term and an
    instruction term:

        similarity = CFG_ALPHA * sim_struct + (1 - CFG_ALPHA) * instr_term

    where:

      1. sim_struct : block-structure similarity from the Hungarian block GED
                      (cfg_block_ged_similarity) = 1 - D_ged / (|V1|+|E1|+|V2|+|E2|).
                      Penalises added / removed / relabeled blocks and their edges.
      2. instr_term : size-weighted (pooled) instruction similarity over the MATCHED
                      block pairs only. For each matched pair we align instructions
                      with Needleman-Wunsch, then pool:

                          instr_term = 1 - (sum d_i) / (sum max(len_i))

                      d_i is the NW instruction distance for matched pair i and
                      max(len_i) its larger instruction count. This is exactly a
                      size-weighted mean of the per-block sims (weight = max(len_i)),
                      so a large divergent block outweighs a trivial one -- the fix
                      for the old unweighted-mean blind spot.

    Unmatched blocks (seed-only "deleted", mutant-only "inserted") are NOT counted in
    instr_term -- sim_struct already accounts for them, so this avoids a double
    penalty. The old caveat (a mutant carrying whole extra blocks reading 1.0) no
    longer applies: those blocks now pull sim_struct down.

    Returns dict:
        similarity     the blend above
        sim_struct     the structural term
        instr_term     the pooled instruction term
        per_block      [(ref_id, mut_id, distance, max_len, block_sim), ...]
        matched        [(ref_id, mut_id), ...]      from the block GED
        deleted        [ref_block_id, ...]          seed-only blocks
        inserted       [mut_block_id, ...]          mutant-only blocks
        block_ged      alias of sim_struct (kept for existing callers)
        n_scored       how many matched pairs went into instr_term
        instrs_dropped (n_from_deleted, n_from_inserted) instructions not in instr_term
    """
    ged = cfg_block_ged_similarity(g1, g2)
    seqs1 = cfg_block_instruction_sequences(g1)
    seqs2 = cfg_block_instruction_sequences(g2)

    per_block = []
    total_distance = 0
    total_max_len = 0
    for rid, mid in ged["matched"]:
        rseq = seqs1.get(rid, [])
        mseq = seqs2.get(mid, [])
        _a_ref, _a_mut, distance = needleman_wunsch(rseq, mseq, sub, gap)
        max_len = max(len(rseq), len(mseq))
        # Both blocks empty -> nothing to disagree about, same convention as align().
        block_sim = 1.0 if max_len == 0 else 1 - distance / max_len
        per_block.append((rid, mid, distance, max_len, block_sim))
        total_distance += distance
        total_max_len += max_len

    # Pooled / size-weighted instruction term over matched pairs only.
    if total_max_len > 0:
        instr_term = 1 - total_distance / total_max_len
    elif not (g1.get("BLOCKS") or g2.get("BLOCKS")):
        instr_term = 1.0        # two blockless CFGs: nothing to disagree about
    else:
        instr_term = 0.0        # blocks exist but no matched instructions to compare

    # Safety net only: cfg_block_ged_similarity now normalizes by the true max edit
    # distance (|V1|+2|E1|+|V2|+2|E2|), so ged["similarity"] is already in [0,1] and
    # this max() is inert. It used to rescue a negative value from the old under-
    # counted denominator. block_ged carries the same value for diagnostics.
    sim_struct = max(0.0, ged["similarity"])
    similarity = CFG_ALPHA * sim_struct + (1 - CFG_ALPHA) * instr_term

    instrs_dropped = (
        sum(len(seqs1.get(rid, [])) for rid in ged["deleted"]),
        sum(len(seqs2.get(mid, [])) for mid in ged["inserted"]),
    )

    return {
        "similarity": similarity,
        "sim_struct": sim_struct,
        "instr_term": instr_term,
        "per_block": per_block,
        "matched": ged["matched"],
        "deleted": ged["deleted"],
        "inserted": ged["inserted"],
        "block_ged": ged["similarity"],
        "n_scored": len(per_block),
        "instrs_dropped": instrs_dropped,
    }


def cfg_ged_nw_similarity(g1, g2, sub=SUB_COST, gap=GAP_COST):
    """
    Two-level CFG similarity: match blocks with the Hungarian block GED
    (cfg_block_ged_similarity), then align each matched block pair's instructions
    with Needleman-Wunsch, and fold both levels into one instruction-level edit
    distance:

        total_distance = sum over matched block pairs of the NW instruction distance
                       + every instruction in an unmatched (deleted/inserted) block
        similarity     = 1 - total_distance / max(total_instr(g1), total_instr(g2))

    Each instruction is charged exactly once: a matched block contributes only the
    instructions NW could not align, an unmatched block is a full indel. Two
    instruction-free CFGs are defined as identical (1.0).

    CAVEAT: the block GED matches blocks on topology (predecessor/successor
    structure), not content, so it can pair a large block with a near-empty one.
    When it does, the matched-pair NW distances can exceed max(total1, total2) and
    drive the raw score below 0, so the result is clamped to [0, 1]. A score stuck
    at 0.0 is a hint the topology matching is pairing very differently sized blocks.

    Returns dict: similarity, distance, denominator, matched, deleted, inserted,
    per_block [(ref_id, mut_id, block_distance, block_max_len), ...].
    """
    ged = cfg_block_ged_similarity(g1, g2)
    instr1 = cfg_instructions_by_block(g1)
    instr2 = cfg_instructions_by_block(g2)

    total_distance = 0
    per_block = []
    for rid, mid in ged["matched"]:
        rseq = instr1.get(rid, [])
        mseq = instr2.get(mid, [])
        _a_ref, _a_mut, dist = needleman_wunsch(rseq, mseq, sub, gap)
        total_distance += dist
        per_block.append((rid, mid, dist, max(len(rseq), len(mseq))))

    # Unmatched blocks: every instruction is an indel (deletion or insertion).
    for rid in ged["deleted"]:
        total_distance += len(instr1.get(rid, []))
    for mid in ged["inserted"]:
        total_distance += len(instr2.get(mid, []))

    # Totals over the GED-covered block sets (matched + deleted == all g1 blocks;
    # matched + inserted == all g2 blocks) so every instruction is counted once.
    total1 = (sum(len(instr1.get(r, [])) for r, _ in ged["matched"])
              + sum(len(instr1.get(r, [])) for r in ged["deleted"]))
    total2 = (sum(len(instr2.get(m, [])) for _, m in ged["matched"])
              + sum(len(instr2.get(m, [])) for m in ged["inserted"]))
    denominator = max(total1, total2)

    if denominator == 0:
        similarity = 1.0
    else:
        similarity = max(0.0, 1.0 - total_distance / denominator)

    return {
        "similarity": similarity,
        "distance": total_distance,
        "denominator": denominator,
        "matched": ged["matched"],
        "deleted": ged["deleted"],
        "inserted": ged["inserted"],
        "per_block": per_block,
    }


# ---------------------------------------------------------------------------
# Block-to-block similarity via a simplified LLVM IR-similarity check.
#
# Two blocks are "similar" iff they perform the same series of operations for all
# inputs: same opcode at each position AND operands/results that map under a
# single consistent value bijection between the blocks (LLVM's structural check).
# Output is binary -- 1.0 if structurally identical, else 0.0.
#
# Simplifications vs real LLVM: opcode identity is the node title only; operand
# order is taken from EDGES order; commutative operations are not reordered.
# ---------------------------------------------------------------------------

def cfg_block_instructions(graph):
    """
    Per block, the ordered list of (node_id, opcode, operand_value_ids).

    opcode  = node title.
    operands = sources of this node's incoming data-flow edges (EDGES with
               to == node), in EDGES order -- our proxy for LLVM operand position.
    Instructions within a block are ordered by node id.
    """
    operands = defaultdict(list)
    for e in graph.get("EDGES", []) or []:
        if isinstance(e, dict) and e.get("to") is not None:
            operands[e["to"]].append(e.get("from"))

    by_block = defaultdict(list)
    nodes = [n for n in graph.get("NODES", []) or [] if isinstance(n, dict)]
    for n in sorted(nodes, key=lambda n: (n.get("id") is None, n.get("id"))):
        nid = n.get("id")
        by_block[n.get("block_id")].append(
            (nid, n.get("title"), tuple(operands.get(nid, [])))
        )
    return by_block


def llvm_blocks_identical(instrs_a, instrs_b):
    """
    True iff two instruction sequences are structurally identical in the
    simplified LLVM sense: same length, same opcode at each position, and every
    operand/result maps under one consistent value bijection between the blocks.

    instrs_* are lists of (node_id, opcode, operand_value_ids) from
    cfg_block_instructions.
    """
    if len(instrs_a) != len(instrs_b):
        return False

    fwd, bwd = {}, {}  # A-value <-> B-value bijection

    def unify(x, y):
        if fwd.get(x, y) != y or bwd.get(y, x) != x:
            return False
        fwd[x] = y
        bwd[y] = x
        return True

    for (aid, aop, aoperands), (bid, bop, boperands) in zip(instrs_a, instrs_b):
        if aop != bop or len(aoperands) != len(boperands):
            return False
        for x, y in zip(aoperands, boperands):
            if not unify(x, y):
                return False
        if not unify(aid, bid):   # results are values too, used by later instrs
            return False
    return True


def cfg_block_llvm_similarity(graph_a, block_a, graph_b, block_b):
    """
    Binary block-to-block similarity (1.0 / 0.0) between block_a of graph_a and
    block_b of graph_b, via the simplified LLVM IR-similarity structural check.
    """
    a = cfg_block_instructions(graph_a).get(block_a, [])
    b = cfg_block_instructions(graph_b).get(block_b, [])
    return 1.0 if llvm_blocks_identical(a, b) else 0.0


def load_snapshots(mutation_dir):
    """
    Walk a mutation's extracted directory and return its per-phase graph
    snapshots taken AFTER each phase ran.

    Returns:
        {phase_name: {graph_type: graph_dict}}
        where graph_dict is {"NODES": [...], "EDGES": [...], "KIND": [...]}.
    """
    trace = {}
    for entry in sorted(os.listdir(mutation_dir)):
        phase_path = os.path.join(mutation_dir, entry)
        if not os.path.isdir(phase_path) or not entry.startswith("phase_"):
            continue

        phase_name = "_".join(entry.split("_")[2:])
        trace[phase_name] = {}

        for fname in os.listdir(phase_path):
            if not fname.startswith("after_") or not fname.endswith(".json"):
                continue
            graph_type = fname[len("after_"):-len(".json")]
            with open(os.path.join(phase_path, fname)) as f:
                trace[phase_name][graph_type] = json.load(f)

    return trace


_EMPTY_GRAPH = {"NODES": [], "EDGES": []}


def _cell_is_cfg(g1, g2):
    """
    Is this graph-type cell a Turboshaft CFG (score with the block-mean) or a
    sea-of-nodes graph (score with the flat Needleman-Wunsch)? Decide from
    whichever side actually has content, so a phase one program skipped (empty
    graph) doesn't misclassify the cell. Two empty graphs -> not CFG (the SON
    path's align() already defines empty-vs-empty as 1.0).
    """
    for g in (g1, g2):
        if g.get("NODES") or g.get("BLOCKS"):
            return graph_kind(g) == CFG_KIND
    return False


def score_cell(g1, g2, sub=SUB_COST, gap=GAP_COST):
    """
    Similarity of one (phase, graph-type) cell between two graphs.

    CFG (Turboshaft): a blend of a structural term and an instruction term --
    match blocks (Hungarian block GED) for sim_struct, align each matched block
    pair's instructions with Needleman-Wunsch and pool them size-weighted for
    instr_term, then combine as CFG_ALPHA*sim_struct + (1-CFG_ALPHA)*instr_term
    (cfg_block_mean_similarity). This is the CFG metric of record; the flat
    whole-graph NW is no longer used to score CFG cells.

    SON (sea-of-nodes): the flat whole-graph Needleman-Wunsch over the node
    sequence (build_sequence + align), over (ident, degree) node tokens.
    """
    if _cell_is_cfg(g1, g2):
        return cfg_block_mean_similarity(g1, g2, sub, gap)["similarity"]
    return align(build_sequence(g1), build_sequence(g2), sub, gap)["similarity"]


def compare_mutant(ref_snap, mut_snap, sub=SUB_COST, gap=GAP_COST):
    """
    Compare one mutant's snapshots against the seed's, per phase per graph type.

      Returns:
        phase_sims:   {phase_name: {graph_type: similarity}}
        global_score: float in [0, 1]
    """
    phase_sims = {}
    cell_sims = []

    # Intersection, not union: a phase only one program ran (e.g. the seed's conditional
    # TFDecompressionOptimization, which 0 mutants schedule) has nothing to compare against.
    # Under a union it was aligned to an empty graph -> similarity 0.0, a phantom score that
    # dragged down every mutant's global. Only phases both programs actually produced are
    # comparable.
    for phase in set(ref_snap) & set(mut_snap):
        rgt = ref_snap.get(phase, {})
        mgt = mut_snap.get(phase, {})
        for gtype in set(rgt) | set(mgt):
            similarity = score_cell(rgt.get(gtype) or _EMPTY_GRAPH,
                                    mgt.get(gtype) or _EMPTY_GRAPH, sub, gap)
            phase_sims.setdefault(phase, {})[gtype] = similarity
            cell_sims.append(similarity)

    global_score = sum(cell_sims) / len(cell_sims) if cell_sims else 1.0
    return phase_sims, global_score


def scan_mutations(extracted_dir, reference_name):
    """
    Scan extracted/passing/ and extracted/failing/ for mutation directories.

    Returns:
        list of (mutation_name, label, mutation_dir)
    """
    mutations = []
    for label in ("passing", "failing"):
        label_dir = os.path.join(extracted_dir, label)
        if not os.path.isdir(label_dir):
            print(f"Warning: {label_dir}/ not found, skipping.")
            continue
        for name in sorted(os.listdir(label_dir)):
            mut_dir = os.path.join(label_dir, name)
            if os.path.isdir(mut_dir):
                mutations.append((name, label, mut_dir))
    return mutations

def format_overview(results):
    """Build the summary-table text (rows = mutations, most-divergent cells)."""
    lines = [
        f"{'Mutation':<22} {'Label':<10} {'Global':>8}  "
        f"{'Most Divergent (phase.type=sim)'}",
        "-" * 100,
    ]
    for name, label, phase_sims, global_score in results:
        cells = [(p, g, s) for p, gm in phase_sims.items() for g, s in gm.items()]
        cells.sort(key=lambda c: c[2])
        divergent = "  ".join(
            f"{p.split('.')[-1]}.{g}={s:.2f}" for p, g, s in cells[:3]
        )
        lines.append(f"{name:<22} {label:<10} {global_score:>8.4f}  {divergent}")
    return "\n".join(lines)


def print_table(results):
    """Print the summary table to terminal and return it, so main can also save it."""
    text = format_overview(results)
    print("\n" + text + "\n")
    return text


def write_csvs(results, phase_order, all_graph_types, output_dir="."):
    """One CSV per graph type: rows = mutations, columns = phases."""
    for gtype in sorted(all_graph_types):
        fname = os.path.join(output_dir, f"similarity_{gtype}.csv")
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["mutation", "label", "avg"] + phase_order)
            for name, label, phase_sims, _global in results:
                vals = [phase_sims.get(p, {}).get(gtype) for p in phase_order]
                present = [v for v in vals if v is not None]
                avg = sum(present) / len(present) if present else None
                row = [name, label, f"{avg:.6f}" if avg is not None else ""]
                row += [f"{v:.6f}" if v is not None else "" for v in vals]
                writer.writerow(row)
        print(f"CSV saved: {fname}")


def write_combined_csv(results, phase_order, output_dir="."):
    fname = os.path.join(output_dir, "similarity_combined.csv")
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mutation", "label", "global"] + phase_order)
        for name, label, phase_sims, global_score in results:
            row = [name, label, f"{global_score:.6f}"]
            for phase in phase_order:
                gtype_sims = list(phase_sims.get(phase, {}).values())
                val = sum(gtype_sims) / len(gtype_sims) if gtype_sims else None
                row.append(f"{val:.6f}" if val is not None else "")
            writer.writerow(row)
    print(f"CSV saved: {fname}")


# Output subdirectories, created under output_dir (bug root, or the current dir in
# positional mode). Three answers to three different questions -- see module docstring.
SIM_TO_SEED_DIR = "similarity_to_seed"      # mutant post-phase graph vs seed's
SIM_PER_PHASE_DIR = "similarity_per_phase"  # before-phase vs after-phase, within a program
SIM_IRS_DIR = "similarity_irs"              # extracted IRs annotated with to-seed similarity


def load_before_after(program_dir):
    """
    Like load_snapshots, but keeps BOTH the before_ and after_ snapshot of every
    phase/graph type, so one phase's own transformation can be measured.

    Returns:
        {phase_name: {graph_type: {"before": graph, "after": graph}}}
    A missing side is simply absent; the caller treats it as an empty graph.
    """
    trace = {}
    for entry in sorted(os.listdir(program_dir)):
        phase_path = os.path.join(program_dir, entry)
        if not os.path.isdir(phase_path) or not entry.startswith("phase_"):
            continue

        phase_name = "_".join(entry.split("_")[2:])
        gts = {}
        for fname in os.listdir(phase_path):
            if not fname.endswith(".json"):
                continue
            if fname.startswith("after_"):
                gtype, side = fname[len("after_"):-len(".json")], "after"
            elif fname.startswith("before_"):
                gtype, side = fname[len("before_"):-len(".json")], "before"
            else:
                continue
            with open(os.path.join(phase_path, fname)) as f:
                gts.setdefault(gtype, {})[side] = json.load(f)
        trace[phase_name] = gts

    return trace


def phase_delta_similarities(program_dir, sub=SUB_COST, gap=GAP_COST):
    """
    For ONE program (the seed or a mutant), align S- (before the phase) against
    S+ (after the phase) within each optimization phase, per graph type.

        similarity 1.0   -> the phase left this graph untouched
        similarity < 1.0 -> the phase transformed it (lower = more change)

    This is intra-program: it never looks at the seed's graphs, so it answers
    "how much work did this phase do" rather than "how far did this mutant drift."

    Returns:
        {phase_name: {graph_type: similarity}}
    """
    deltas = {}
    for phase, gts in load_before_after(program_dir).items():
        deltas[phase] = {}
        for gtype, sides in gts.items():
            deltas[phase][gtype] = score_cell(sides.get("before") or _EMPTY_GRAPH,
                                               sides.get("after") or _EMPTY_GRAPH, sub, gap)
    return deltas


def write_per_phase_csvs(program_deltas, phase_order, all_graph_types, output_dir):
    """
    similarity_per_phase/: one CSV per graph type + a combined CSV. Rows = programs
    (seed first, then every mutant), columns = phases in execution order. Each cell is
    the before->after similarity of that phase for that program.

    program_deltas: ordered list of (program_name, label, {phase: {gtype: similarity}}).
    """
    for gtype in sorted(all_graph_types):
        fname = os.path.join(output_dir, f"similarity_{gtype}.csv")
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["program", "label", "avg"] + phase_order)
            for name, label, deltas in program_deltas:
                vals = [deltas.get(p, {}).get(gtype) for p in phase_order]
                present = [v for v in vals if v is not None]
                avg = sum(present) / len(present) if present else None
                row = [name, label, f"{avg:.6f}" if avg is not None else ""]
                row += [f"{v:.6f}" if v is not None else "" for v in vals]
                writer.writerow(row)
        print(f"CSV saved: {fname}")

    fname = os.path.join(output_dir, "similarity_combined.csv")
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["program", "label", "avg"] + phase_order)
        for name, label, deltas in program_deltas:
            per_phase = []
            for p in phase_order:
                sims = list(deltas.get(p, {}).values())
                per_phase.append(sum(sims) / len(sims) if sims else None)
            present = [v for v in per_phase if v is not None]
            avg = sum(present) / len(present) if present else None
            row = [name, label, f"{avg:.6f}" if avg is not None else ""]
            row += [f"{v:.6f}" if v is not None else "" for v in per_phase]
            writer.writerow(row)
    print(f"CSV saved: {fname}")


def write_ir_with_similarities(mutations, phase_sims_by_name, prior_sims_by_name, output_dir):
    """
    similarity_irs/: mirror each mutant's extracted tree, writing every scored
    after_<gtype>.json with two added top-level keys, in this order:

        "baseline_similarity"    -- this (phase, graph type) cell's similarity to
                                    the seed's post-phase graph (cross-program drift).
        "prior_phase_similarity" -- the same graph's before->after similarity within
                                    this program (how much this phase changed it).

    before_ files are skipped (never scored to the seed) and so is the reference
    itself (it is the baseline, not scored against itself).
    """
    count = 0
    for name, label, mut_dir in mutations:
        phase_sims = phase_sims_by_name.get(name, {})
        prior_sims = prior_sims_by_name.get(name, {})
        for entry in sorted(os.listdir(mut_dir)):
            phase_path = os.path.join(mut_dir, entry)
            if not os.path.isdir(phase_path) or not entry.startswith("phase_"):
                continue
            phase_name = "_".join(entry.split("_")[2:])
            for fname in os.listdir(phase_path):
                if not fname.startswith("after_") or not fname.endswith(".json"):
                    continue
                gtype = fname[len("after_"):-len(".json")]
                with open(os.path.join(phase_path, fname)) as f:
                    graph = json.load(f)
                graph["baseline_similarity"] = phase_sims.get(phase_name, {}).get(gtype)
                graph["prior_phase_similarity"] = prior_sims.get(phase_name, {}).get(gtype)
                out_phase = os.path.join(output_dir, label, name, entry)
                os.makedirs(out_phase, exist_ok=True)
                with open(os.path.join(out_phase, fname), "w") as f:
                    json.dump(graph, f, indent=2)
                count += 1
    print(f"Annotated IRs written under: {output_dir}  ({count} files)")


def main():
    parser = argparse.ArgumentParser(
        description="Align seed vs mutant IRs (id-ordered sequences + Needleman-Wunsch)."
    )
    parser.add_argument("--bug",
                        help="Bug name; uses bugs/<bug>/extracted and writes CSVs to bugs/<bug>/")
    parser.add_argument("extracted_dir", nargs="?",
                        help="Path to the extracted/ directory (alternative to --bug)")
    parser.add_argument("--reference", default="turbo-buggy",
                        help="Reference (seed) subdirectory name (default: turbo-buggy)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to write CSVs into (default: bug root, or current dir)")
    args = parser.parse_args()

    if args.bug:
        P = bug_paths(args.bug)
        extracted_dir = P.extracted
        output_dir = args.output_dir or P.root
    elif args.extracted_dir:
        extracted_dir = args.extracted_dir
        output_dir = args.output_dir or "."
    else:
        parser.error("provide --bug <name> or a positional extracted_dir")

    reference_dir = os.path.join(extracted_dir, args.reference)
    if not os.path.isdir(reference_dir):
        print(f"Error: reference directory not found: {reference_dir}")
        sys.exit(1)

    print(f"Loading seed snapshots from: {reference_dir}")
    ref_snap = load_snapshots(reference_dir)
    print(f"  {len(ref_snap)} phases found in seed.")

    mutations = scan_mutations(extracted_dir, args.reference)
    if not mutations:
        print("No mutations found in extracted/passing/ or extracted/failing/.")
        sys.exit(1)

    print(f"Found {len(mutations)} mutations. Aligning...\n")

    results = []
    all_graph_types = set()
    for name, label, mut_dir in mutations:
        mut_snap = load_snapshots(mut_dir)
        phase_sims, global_score = compare_mutant(ref_snap, mut_snap)
        for gm in phase_sims.values():
            all_graph_types.update(gm)
        results.append((name, label, phase_sims, global_score))

    # Phase order = execution order from the seed directory.
    phase_order = [
        "_".join(e.split("_")[2:])
        for e in sorted(os.listdir(reference_dir))
        if os.path.isdir(os.path.join(reference_dir, e)) and e.startswith("phase_")
    ]
    seen = set(phase_order)
    for _, _, phase_sims, _ in results:
        for p in phase_sims:
            if p not in seen:
                phase_order.append(p)
                seen.add(p)

    overview = print_table(results)
    overview_path = os.path.join(output_dir, "overview.txt")
    with open(overview_path, "w") as f:
        f.write(f"Reference: {reference_dir}\n"
                f"Phases in seed: {len(ref_snap)}\n"
                f"Mutations aligned: {len(mutations)}\n\n"
                f"{overview}\n")
    print(f"Overview saved: {overview_path}")

    # Three sibling output directories, each answering a different question.
    to_seed_dir = os.path.join(output_dir, SIM_TO_SEED_DIR)
    per_phase_dir = os.path.join(output_dir, SIM_PER_PHASE_DIR)
    irs_dir = os.path.join(output_dir, SIM_IRS_DIR)
    for d in (to_seed_dir, per_phase_dir, irs_dir):
        os.makedirs(d, exist_ok=True)

    # 1. similarity_to_seed/ -- mutant post-phase graph vs seed's (passing rows only).
    write_csvs(results, phase_order, all_graph_types, to_seed_dir)
    write_combined_csv(results, phase_order, to_seed_dir)

    # 2. similarity_per_phase/ -- before-phase vs after-phase, for seed AND every mutant.
    program_deltas = [("seed", "seed", phase_delta_similarities(reference_dir))]
    for name, label, mut_dir in mutations:
        program_deltas.append((name, label, phase_delta_similarities(mut_dir)))
    pp_graph_types = set()
    for _, _, deltas in program_deltas:
        for gm in deltas.values():
            pp_graph_types.update(gm)
    write_per_phase_csvs(program_deltas, phase_order, pp_graph_types, per_phase_dir)

    # 3. similarity_irs/ -- extracted IRs annotated with baseline (to-seed) and
    #    prior-phase (before->after) similarity.
    phase_sims_by_name = {name: ps for name, _label, ps, _g in results}
    prior_sims_by_name = {name: deltas for name, _label, deltas in program_deltas}
    write_ir_with_similarities(mutations, phase_sims_by_name, prior_sims_by_name, irs_dir)


if __name__ == "__main__":
    main()
