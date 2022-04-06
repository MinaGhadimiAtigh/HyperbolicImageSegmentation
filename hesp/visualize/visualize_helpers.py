import math
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from colorspace.colorlib import HCL as HCL_lib
from matplotlib.colors import ListedColormap, to_rgba

from hesp.hierarchy.hierarchy_helpers import hierarchy_pos, json2rels


def save_metrics(config, loss_hist, acc_hist, curv_hist):
    fig = plt.figure()
    plt.plot(loss_hist, label='loss')
    plt.plot(curv_hist, label='curvature')
    plt.plot(acc_hist, label='accuracy')
    plt.legend()
    fig.savefig(os.path.join(config._PROTOTYPER_SAVE_DIR, 'metrics.png'), dpi=100)


def cord2pix(coord, max_norm, resolution):
    """ Converts coordinates to the correct pixel locations in the final plot
    Args:
        coord: x,y coordinates of point
        max_norm: bounds of x,y coordinates
    Returns:
        (r,c): pixel position of coord
    """
    return ((resolution / (2 * max_norm)) * coord) + (resolution / 2)


def draw_hierch(rels, start):
    # This function can be used to draw the hierarchies and save them
    # TODO add a new func to save hierarchies
    G = nx.Graph()
    G.add_edges_from(rels)
    fig = plt.figure(figsize=(70, 70))
    main_ax = fig.subplots()
    main_ax.set_aspect('equal')
    main_ax.axis('off')
    pos = hierarchy_pos(G, start, vert_gap=0.27, vert_loc=-3, width=2, leaf_vs_root_factor=1)
    nx.draw(G, ax=main_ax, pos=pos, with_labels=False, edge_color='lightgrey', node_color='lightblue', node_size=90)

    description = nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes})
    for node, t in description.items():
        t.set_rotation(-90)
        (x, y) = pos[node]
        new_pos = (x, y - .0025)
        t.set_va('top')
        t.set_ha('center')
        t.set_position(new_pos)
        t.set_size(40)


def draw_hierch_rad(rels, start):
    """Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723 """
    G = nx.Graph()
    G.add_edges_from(rels)
    fig = plt.figure(figsize=(30, 30))
    main_ax = fig.subplots()
    main_ax.set_aspect('equal')
    main_ax.axis('off')
    pos = hierarchy_pos(G, start, width=1 * math.pi, vert_gap=1.)
    new_pos = {
        u: (r * math.cos(theta), r * math.sin(theta)) for u, (theta, r) in pos.items()
    }
    nx.draw(
        G, ax=main_ax, pos=new_pos, node_size=5, with_labels=True, font_size=10,
    )
    nx.draw_networkx_nodes(
        G, ax=main_ax, pos=new_pos, nodelist=[start], node_color="blue", node_size=5
    )
    plt.show()


def draw_hierch_from_json(json, radial=False):
    rels = json2rels(json)
    start = list(json.keys())[0]
    if radial:
        draw_hierch_rad(rels, start)
    else:
        draw_hierch(rels, start)


def plot_hierarchy(tree, ax, show_idx):
    """ Display hierarchy in a given axis. """
    show_names = [tree.i2n[idx] for idx in show_idx]
    edges = [e for e in tree.G.edges() if (e[0] in show_names + [tree.root]) and (e[1] in show_names + [tree.root])]
    nx.draw(
        tree.G, ax=ax, pos=tree.pos, node_size=10, aplha=0, with_labels=False, width=0.5, edge_color='#f0f0f0',
        node_color='#f0f0f0', font_size=2,
    )
    nx.draw(
        tree.G, ax=ax, pos=tree.pos, node_size=10, edgelist=edges, nodelist=show_names, node_color='darkgrey',
        with_labels=False, width=0.5, edge_color='grey',
    )
    nx.draw_networkx_nodes(
        tree.G, pos=tree.pos, nodelist=[tree.root], node_color="orange", node_size=20
    )


def hcl2hex(H, C, L):
    HCL_color = HCL_lib(H=H, C=C, L=L)
    HCL_color.to('hex', fixup=True)  # convert to hex
    return HCL_color.get()['hex_'][0]


def make_alpha_map(c):
    rgba = to_rgba(c)

    my_cmap = np.stack([rgba for i in range(256)])

    # Set alpha
    my_cmap[:, -1] = np.linspace(0, 1, 256)

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)
    return my_cmap


def colour_nodes(nodes, root_name):
    """ Somewhat crude implementation of a hierarchical color scheme proposed by
    'Tree Colors: Color Schemes for Tree-Structured Data', by Martijn Tennekes and Edwin de Jonge.
    Approximation where:
    hue color for branch/ width
    chroma increase with depth
    luminance decrease with depth.
    Args:
        nodes: dictionary of concept_name : Node objects
        root_name: name of root concept
    Returns:
        nodes: where the Node objects have HCL colour assigned.
    """
    root = nodes[root_name]
    f = 0.90
    max_depth = np.max([n.depth for _, n in nodes.items()])
    depth2chroma = {i: chroma for i, chroma in enumerate(np.linspace(60, 80, max_depth + 1))}
    depth2luminance = {i: lum for i, lum in enumerate(np.linspace(95, 50, max_depth + 1))}

    def assign_colours(parent):
        H = np.linspace(parent.hue_range[0], parent.hue_range[1], 2, endpoint=False)[-1]
        H = (H - 50) % 361
        C = depth2chroma[parent.depth]
        L = depth2luminance[parent.depth]
        parent.hex = hcl2hex(H, C, L)
        parent.HCL = (H, C, L)
        r = parent.hue_range[1] - parent.hue_range[0]  # width
        fractor = r * (1 - f) // 2
        hue_start = parent.hue_range[0] + fractor
        hue_end = parent.hue_range[1] - fractor
        hues = np.linspace(hue_start, hue_end, len(parent.children) + 1)
        for i, c in enumerate(parent.children):
            child_node = nodes[c]
            child_node.hue_range = (hues[i], hues[i + 1])
            assign_colours(child_node)

    root.hue_range = (0, 360)
    assign_colours(root)

    return nodes
