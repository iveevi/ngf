import bpy

from typing import Tuple


def clean_nodes(nodes: bpy.types.Nodes) -> None:
    for node in nodes:
        nodes.remove(node)


def add_material(name: str = 'Material',
                 use_nodes: bool = False,
                 make_node_tree_empty: bool = False) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = use_nodes
    if use_nodes and make_node_tree_empty:
        clean_nodes(material.node_tree.nodes)

    return material


def set_principled_node(principled_node: bpy.types.Node,
                        base_color: Tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0),
                        metallic: float = 0.0,
                        roughness: float = 0.5,
                        ior: float = 1.45,
                        transmission: float = 0.0) -> None:
    principled_node.inputs['Base Color'].default_value = base_color
    principled_node.inputs['Metallic'].default_value = metallic
    principled_node.inputs['Roughness'].default_value = roughness
    principled_node.inputs['Transmission Weight'].default_value = transmission
    principled_node.inputs['IOR'].default_value = ior

