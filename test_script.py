# from pxr import Usd, UsdGeom
# import omni

# stage = omni.usd.get_context().get_stage()
# xformPrim = UsdGeom.Xform.Define(stage, '/hello')
# spherePrim = UsdGeom.Sphere.Define(stage, '/hello/world')

from omni.isaac.core import World
from pxr import Usd, UsdGeom
import omni

# Create default plane
world = World()
world.scene.add_default_ground_plane()

# Add a sphere
stage = omni.usd.get_context().get_stage()
xformPrim = UsdGeom.Xform.Define(stage, "/hello")
spherePrim = UsdGeom.Sphere.Define(stage, "/hello/sphere")

world.clear()

world.initialize_physics()


# add_object.py
import numpy as np
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core import World

# Instantiate World class
world = World()

# Add cuboid to the stage
target_cuboid = DynamicCuboid(
    prim_path="/World/Target",  # Path in the stage
    name="target_cuboid",  # Nickname in the scene
    position=np.array([0.6, 0.6, 0.1]),  # Initial position as an array [x, y, z]
    orientation=np.array(
        [1, 0, 0, 0]
    ),  # Initial orientation as an array [qw, qx, qy, qz]
    color=np.array([1, 0, 0]),  # Normalized RGB color (red)
    size=0.05,  # Size of the cuboid, size*scale = [length, width, height]
    scale=np.array([1, 1, 1]),  # Scale factors for the cuboid
    mass=0.05,  # Mass of the cuboid in kilograms
)

# Register the cuboid in the scene
world.scene.add(target_cuboid)

world.initialize_physics()

# Reset the world to reinitialize objects in the scene
world.reset_async()
