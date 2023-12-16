import polyscope as ps
import meshio

target = meshio.read("target.obj")
simplified = meshio.read("simplified.obj")
quadrangulated = meshio.read("quadrangulated.obj")

ps.init()
ps.set_ground_plane_mode("none")
ps.register_surface_mesh("target", target.points, target.cells_dict["triangle"]).set_color([0.5, 0.5, 1.0])
ps.register_surface_mesh("simplified", simplified.points, simplified.cells_dict["triangle"]).set_color([0.5, 0.5, 1.0])
ps.register_surface_mesh("quadrangulated", quadrangulated.points, quadrangulated.cells_dict["quad"]).set_color([0.5, 0.5, 1.0])
ps.show()
