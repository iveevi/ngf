import bpy
import math

from mathutils import Vector
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
                        coat: float = 1.0,
                        coat_roughness: float = 0.075,
                        coat_ior: float = 1.7) -> None:
    principled_node.inputs['Base Color'].default_value = base_color
    principled_node.inputs['Metallic'].default_value = metallic
    principled_node.inputs['Roughness'].default_value = roughness
    principled_node.inputs['Coat Weight'].default_value = coat
    principled_node.inputs['Coat Roughness'].default_value = coat_roughness
    principled_node.inputs['Coat IOR'].default_value = coat_ior

def normalize(M: bpy.types.Mesh) -> None:
    vertices = M.data.vertices
    matrix = M.matrix_world
    vertices = [matrix @ v.co for v in vertices]

    vmin, vmax = 0, 0
    vecmin = Vector((0, 0, 0))
    vecmax = Vector((0, 0, 0))

    for v in vertices:
        vmax = max(vmax, v.x)
        vmax = max(vmax, v.y)
        vmax = max(vmax, v.z)

        vmin = min(vmin, v.x)
        vmin = min(vmin, v.y)
        vmin = min(vmin, v.z)

        vecmax.x = max(vecmax.x, v.x)
        vecmax.y = max(vecmax.y, v.y)
        vecmax.z = max(vecmax.z, v.z)

        vecmin.x = max(vecmin.x, v.x)
        vecmin.y = max(vecmin.y, v.y)
        vecmin.z = max(vecmin.z, v.z)

    center = (vecmin + vecmin)/2
    scale = abs(vmax - vmin)/2

    for v in M.data.vertices:
        v.co = (v.co - center)/scale
