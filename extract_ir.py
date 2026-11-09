"""
The Raw TurboFan IR JSON is dense and unstructured for analysis purposes. One JSON file
mixes control flow, data flow, and effect edges all in one file. The file also
contains an overload of attributes on each node that are not needed for graph comparison
Every optimization phase is also in the one file, so separating them out will help
significantly by allowing for phase-by-phase comparison.

A --trace-turbo dump is NOT one representation. Each phase declares its own kind in
phase["type"]:

  "graph"                  TurboFan sea-of-nodes. data.{nodes, edges}; every edge
                           carries a "type" (value/control/effect/context/frame-state),
                           which is the axis we split on. No blocks.
  "turboshaft_graph"       Turboshaft CFG. data.{nodes, edges, blocks}. Edges are bare
                           {source, target} with NO "type", so the split above does not
                           apply and would silently yield zero edges. Blocks carry the
                           control flow: {id, type, predecessors}.
  "turboshaft_custom_data" Not a graph -- a per-operation annotation table
                           ({key: op_id, value: str}) that describes the MOST RECENT
                           turboshaft_graph phase. Attached positionally.
  everything else          schedule / sequence / instructions / disassembly: text or
                           backend payloads with no node graph. Skipped.

SON phases are written out as before_<gtype>.json / after_<gtype>.json. CFG phases are
written out as before_cfg.json / after_cfg.json, carrying BLOCKS alongside NODES/EDGES.
Both record GRAPH_KIND so similarity.py can dispatch without re-reading the raw dump.
"""

import json
import os
import sys
from collections import Counter

from analyze_ir import extract_v8_info

OUTPUT_DIR = "extracted"

SON_KIND = "graph"
CFG_KIND = "turboshaft_graph"
CUSTOM_KIND = "turboshaft_custom_data"

# Turboshaft operation fields worth keeping. "origin"/"sourcePosition" are deliberately
# excluded from the identity used by similarity.py (origin.nodeId shifts with any
# renumbering upstream), but are kept here for traceability back to the SON node.
CFG_ATTRS = ("id", "title", "block_id", "op_effects", "origin", "sourcePosition")

# turboshaft_custom_data phase name -> the node attribute it becomes after the join.
CUSTOM_DATA_ATTRS = {
    "Properties": "properties",
    "Representations": "representation",
    "Types": "type",
    "Use Count (saturated)": "use_count",
}

def load(path):
    with open(path) as f:
        return json.load(f)

def save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def extract_graph(phase_data, attributes):
    '''
    extract clean graph structures. Separate nodes from edges from kinds
    in a dictionary. This creates a standard dirrected multigraph which
    can more easily be analyzed. System type changes and control flow
    changes are looked at independently. Where a type error appears in
    data flow, only dead code is shown in control flow. By separating
    graphs, you can identify that the problem originated in the DFG (type
    system) but manifested in the CFG (control flow optimization).
    If they were mixed, this would be harder to see.
    '''
    if not isinstance(phase_data, dict):
        return {"NODES": [], "EDGES": [], "KIND": []}

    raw_nodes = phase_data.get("nodes", [])
    raw_edges = phase_data.get("edges", [])

    NODES = []
    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        NODES.append({attr: node.get(attr) for attr in attributes})

    EDGES = []
    KIND = set()
    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        kind = edge.get("type")
        EDGES.append({
            "from": edge.get("source"),
            "to":   edge.get("target"),
            "kind": kind
        })
        if kind:
            KIND.add(kind)

    return {
        "NODES": NODES,
        "EDGES": EDGES,
        "KIND":  sorted(list(KIND))
    }


def extract_cfg_graph(phase_data, custom_data=None):
    '''
    Extract one Turboshaft (CFG) phase into the same NODES/EDGES shape the SON path
    produces, plus BLOCKS.

    Unlike sea-of-nodes, Turboshaft is already scheduled: every operation names the
    basic block it lives in (block_id), and blocks name their predecessors. So the
    control flow lives in BLOCKS, not in an edge attribute -- there is nothing to
    split on, and one CFG graph is emitted per phase rather than one per edge type.

    custom_data maps attribute name -> {op_id: value}, joined onto each operation.
    That is where Turboshaft keeps the attributes SON stores inline on the node
    (properties, representation, type), so without this join the operations carry
    almost no identity.
    '''
    if not isinstance(phase_data, dict):
        return {"NODES": [], "EDGES": [], "BLOCKS": [], "GRAPH_KIND": CFG_KIND}

    custom_data = custom_data or {}

    NODES = []
    for op in phase_data.get("nodes", []):
        if not isinstance(op, dict):
            continue
        node = {attr: op.get(attr) for attr in CFG_ATTRS}
        for attr, by_id in custom_data.items():
            node[attr] = by_id.get(op.get("id"))
        NODES.append(node)

    # Turboshaft edges are bare {source, target}: no "type" to record. kind is kept as
    # None so the dict shape matches the SON path (compute_change reads "kind").
    EDGES = [
        {"from": e.get("source"), "to": e.get("target"), "kind": None}
        for e in phase_data.get("edges", [])
        if isinstance(e, dict)
    ]

    BLOCKS = [
        {
            "id": b.get("id"),
            "kind": b.get("type"),
            "predecessors": b.get("predecessors", []),
        }
        for b in phase_data.get("blocks", [])
        if isinstance(b, dict)
    ]

    return {"NODES": NODES, "EDGES": EDGES, "BLOCKS": BLOCKS, "GRAPH_KIND": CFG_KIND}


def collect_custom_data(phase):
    '''
    Turn one turboshaft_custom_data phase into (attr_name, {op_id: value}).

    These phases have no link back to the graph they describe -- they simply FOLLOW it
    in the phase list -- so the caller is responsible for attaching them to the most
    recent turboshaft_graph phase. Returns (None, None) for tables we do not model.
    '''
    attr = CUSTOM_DATA_ATTRS.get(phase.get("name"))
    if attr is None or phase.get("data_target") != "operations":
        return None, None

    by_id = {}
    for entry in phase.get("data", []) or []:
        if isinstance(entry, dict) and "key" in entry:
            by_id[entry["key"]] = entry.get("value")
    return attr, by_id


def separate_graphs(graph, graph_types):

    nodes     = graph["NODES"]
    all_edges = graph["EDGES"]

    result = {}
    for gtype in graph_types:
        edges = [e for e in all_edges if e.get("kind") == gtype]
        result[gtype] = {
            "NODES": nodes,
            "EDGES": edges,
            "KIND":  [gtype] if edges else []
        }

    return result


def compute_change(before, after):
    # Align nodes by semantic signature (all attributes except id).
    # Using id for alignment is wrong: a mutation that inserts one node early
    # shifts every subsequent id, making identical nodes appear as changes.

    def node_sem_key(node):
        return tuple(sorted((k, str(v)) for k, v in node.items() if k != "id"))

    def node_sem_dict(node):
        return {k: v for k, v in node.items() if k != "id"}

    # Build id→semantic_dict maps so edges can reference node semantics
    before_id_to_sem = {
        n["id"]: node_sem_dict(n)
        for n in before["NODES"]
        if isinstance(n, dict) and n.get("id") is not None
    }
    after_id_to_sem = {
        n["id"]: node_sem_dict(n)
        for n in after["NODES"]
        if isinstance(n, dict) and n.get("id") is not None
    }

    # Multiset comparison: a "modified" node = remove old sem + add new sem
    before_counter = Counter(
        node_sem_key(n) for n in before["NODES"] if isinstance(n, dict)
    )
    after_counter = Counter(
        node_sem_key(n) for n in after["NODES"] if isinstance(n, dict)
    )

    added_keys   = after_counter  - before_counter
    removed_keys = before_counter - after_counter

    # Recover original-typed dicts for JSON output (sem_key has str values)
    after_key_to_dict  = {node_sem_key(n): node_sem_dict(n)
                          for n in after["NODES"]  if isinstance(n, dict)}
    before_key_to_dict = {node_sem_key(n): node_sem_dict(n)
                          for n in before["NODES"] if isinstance(n, dict)}

    added_nodes   = [after_key_to_dict[k]  for k, cnt in added_keys.items()   for _ in range(cnt)]
    removed_nodes = [before_key_to_dict[k] for k, cnt in removed_keys.items() for _ in range(cnt)]

    # Edges: represent endpoints by their semantic dicts, not raw IDs
    def edge_sem_key(edge, id_to_sem):
        from_sem = id_to_sem.get(edge.get("from"))
        to_sem   = id_to_sem.get(edge.get("to"))
        if from_sem is None or to_sem is None:
            return None
        return (
            tuple(sorted((k, str(v)) for k, v in from_sem.items())),
            tuple(sorted((k, str(v)) for k, v in to_sem.items())),
            edge.get("kind")
        )

    before_edge_keys = Counter()
    for e in before["EDGES"]:
        k = edge_sem_key(e, before_id_to_sem)
        if k is not None:
            before_edge_keys[k] += 1

    after_edge_keys = Counter()
    for e in after["EDGES"]:
        k = edge_sem_key(e, after_id_to_sem)
        if k is not None:
            after_edge_keys[k] += 1

    added_edge_keys   = after_edge_keys  - before_edge_keys
    removed_edge_keys = before_edge_keys - after_edge_keys

    def edge_key_to_dict(key):
        from_sem_key, to_sem_key, kind = key
        return {
            "from_sem": dict(from_sem_key),
            "to_sem":   dict(to_sem_key),
            "kind":     kind
        }

    added_edges   = [edge_key_to_dict(k) for k, cnt in added_edge_keys.items()   for _ in range(cnt)]
    removed_edges = [edge_key_to_dict(k) for k, cnt in removed_edge_keys.items() for _ in range(cnt)]

    if not any([added_nodes, removed_nodes, added_edges, removed_edges]):
        return None

    return {
        "added_nodes":   added_nodes,
        "removed_nodes": removed_nodes,
        "added_edges":   added_edges,
        "removed_edges": removed_edges
    }


def collect_phase_records(raw_phases):
    '''
    First pass over the raw phase list: keep the phases that carry a node graph, and
    attach each turboshaft_custom_data table to the turboshaft_graph phase it describes.

    The attachment is positional -- a custom_data phase has no back-reference, it just
    follows its graph -- so it is bound to the most recently seen CFG phase. Getting
    this wrong silently attributes one phase's types/properties to another.
    '''
    records = []
    last_cfg = None

    for i, phase in enumerate(raw_phases):
        if not isinstance(phase, dict):
            continue
        kind = phase.get("type")

        if kind == CUSTOM_KIND:
            attr, by_id = collect_custom_data(phase)
            if attr is not None and last_cfg is not None:
                last_cfg["custom"][attr] = by_id
            continue

        if kind not in (SON_KIND, CFG_KIND):
            continue        # schedule / sequence / instructions / disassembly

        record = {
            "index": i,
            "name": phase.get("name", f"phase_{i}"),
            "kind": kind,
            "data": phase.get("data", {}),
            "custom": {},
        }
        records.append(record)
        if kind == CFG_KIND:
            last_cfg = record

    return records


def son_attributes(raw_phases):
    """Node attributes as they appear on SON phases only (not the whole-file union)."""
    attrs = set()
    for phase in raw_phases:
        if not isinstance(phase, dict) or phase.get("type") != SON_KIND:
            continue
        data = phase.get("data")
        if not isinstance(data, dict):
            continue
        for node in data.get("nodes", []) or []:
            if isinstance(node, dict):
                attrs.update(node.keys())
    return attrs


def extract(input_path, output_dir=OUTPUT_DIR):

    graph_types, _all_attributes, phases = extract_v8_info(input_path)

    data = load(input_path)
    raw_phases = data.get("phases", [])

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    base_name = base_name.replace("-pretty", "").replace("_pretty", "")
    output_base = os.path.join(output_dir, base_name)

    print(f"Extracting: {input_path}")
    print(f"Output: {output_base}/")
    print("...")

    attributes = son_attributes(raw_phases)
    records = collect_phase_records(raw_phases)

    for record in records:
        if record["kind"] == SON_KIND:
            record["graph"] = extract_graph(record["data"], attributes)
        else:
            record["graph"] = extract_cfg_graph(record["data"], record["custom"])

    # A phase's "before" is the previous graph OF THE SAME KIND. The SON pipeline and
    # the CFG pipeline are different representations of the program; diffing the first
    # Turboshaft phase against the last sea-of-nodes phase would report the whole
    # program as changed rather than anything the phase did.
    prev_by_kind = {}

    for record in records:
        i          = record["index"]
        phase_name = record["name"]
        kind       = record["kind"]
        after      = record["graph"]

        dir_name  = f"phase_{i:02d}_{phase_name}"
        phase_dir = os.path.join(output_base, dir_name)

        if kind == SON_KIND:
            before = prev_by_kind.get(SON_KIND, {"NODES": [], "EDGES": [], "KIND": []})
            before_separated = separate_graphs(before, graph_types)
            after_separated  = separate_graphs(after,  graph_types)

            has_changes = False
            for gtype in graph_types:
                save(os.path.join(phase_dir, f"before_{gtype}.json"), before_separated[gtype])
                save(os.path.join(phase_dir, f"after_{gtype}.json"),  after_separated[gtype])
                change = compute_change(before_separated[gtype], after_separated[gtype])
                if change:
                    save(os.path.join(phase_dir, f"change_{gtype}.json"), change)
                    has_changes = True
        else:
            before = prev_by_kind.get(
                CFG_KIND, {"NODES": [], "EDGES": [], "BLOCKS": [], "GRAPH_KIND": CFG_KIND}
            )
            save(os.path.join(phase_dir, "before_cfg.json"), before)
            save(os.path.join(phase_dir, "after_cfg.json"),  after)
            change = compute_change(before, after)
            has_changes = bool(change)
            if change:
                save(os.path.join(phase_dir, "change_cfg.json"), change)

        prev_by_kind[kind] = after

        tag = "CFG" if kind == CFG_KIND else "SON"
        if has_changes:
            print(f"[{i:02d}] {tag}  {phase_name:<46}  CHANGED")
        else:
            print(f"[{i:02d}] {tag}  {phase_name:<46}")

    phase_optimization_mapping = {}
    for record in records:
        phase_optimization_mapping[f"phase_{record['index']:02d}"] = record["name"]

    opt_path = os.path.join(output_base, "phase_optimizations.json")
    save(opt_path, phase_optimization_mapping)

    print("...")
    print(f"Output saved in: {output_base}/")
    print(f"Phase optimization mapping saved in: {opt_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_ir.py <pretty_json_file> [output_dir]")
        print("Example: python extract_ir.py bugs/numbermax/ir_dumps/turbo-buggy-pretty.json bugs/numbermax/extracted/passing")
        sys.exit(1)

    out = sys.argv[2] if len(sys.argv) >= 3 else OUTPUT_DIR
    extract(sys.argv[1], output_dir=out)
