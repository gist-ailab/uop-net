import os
from os.path import join, isdir, isfile
import math
import numpy as np

from pyrep import PyRep
from pyrep.backend import sim
from pyrep.objects import Shape, Dummy


# relative import
if __package__=='' or __package__ is None:
    import sys
    from os import path
    print(path.dirname( path.dirname( path.abspath(__file__) ) ))
    sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
    from file_utils import *
    from timeout import timeout
else:
    from .file_utils import *
    from .timeout import timeout


class BaseEnv:
    def __init__(self, scene_file="", headless=False):
        self.pr = PyRep()
        # Launch the application with a scene file in headless mode
        self.pr.launch(scene_file, headless=headless) 
        
        self.pr.start()    # Start the simulation

    def step(self):
        self.pr.step()

    def stop(self):
        self.pr.stop()
        self.pr.shutdown()

    def load_scene_object_from_file(self, file_path):
        respondable = self.pr.import_model(file_path)
        visible = respondable.get_objects_in_tree(exclude_base=True)[0]
        return SceneObject(respondable_part=respondable, visible_part=visible)

    def load_mesh_from_file(self, file_path, scaling_factor=1):
        shape = Shape.import_shape(filename=file_path,
                                                                         scaling_factor=scaling_factor)
        self.pr.step()
        return shape

class SceneObject(object):
    def __init__(self, respondable_part: Shape, visible_part: Shape, base: Dummy = None):
        self.respondable = respondable_part
        self.visible = visible_part
        name = self.respondable.get_name().replace("_respondable", "")
        try:
            int(name[-1])
            name = name[:-1]
        except:
            pass
        self.name = name

        """Shape Property
        Shapes are collidable, measurable, detectable and renderable objects. This means that shapes:

        collidable: can be used in collision detections against other collidable objects.
        measurable: can be used in minimum distance calculations with other measurable objects.
        detectable: can be detected by proximity sensors.
        renderable: can be detected by vision sensors.
        
        Dynamic shapes will be directly influenced by gravity or other constraints
        Respondable shapes influence each other during dynamic collision

        """
        self._is_collidable = True 
        self._is_measurable = True
        self._is_detectable = True
        self._is_renderable = True
        self._is_dynamic = False
        self._is_respondable = False

        self.initialize_respondable()
        self.initialize_visible()

    def initialize_visible(self):
        self.visible.set_collidable(False)
        self.visible.set_measurable(False)
        self.visible.set_detectable(True)
        self.visible.set_renderable(True)
        self.visible.set_dynamic(False)
        self.visible.set_respondable(False)

    def initialize_respondable(self):
        self.respondable.set_collidable(True)
        self.respondable.set_measurable(True)
        self.respondable.set_detectable(False)
        self.respondable.set_renderable(False)
        self.respondable.set_dynamic(True)
        self.respondable.set_respondable(True)
        self.set_transparency(self.respondable, [0])

    def set_collidable(self, is_collidable):
        self.respondable.set_collidable(is_collidable)
        self._is_collidable = is_collidable

    def set_measurable(self, is_measurable):
        self.respondable.set_measurable(is_measurable)
        self._is_measurable = is_measurable
    
    def set_detectable(self, is_detectable):
        self.visible.set_detectable(is_detectable)
        self._is_detectable = is_detectable
    
    def set_renderable(self, is_renderable):
        self.visible.set_renderable(is_renderable)
        self._is_renderable = is_renderable
    
    def set_respondable(self, is_respondable):
        self.respondable.set_respondable(is_respondable)
        self._is_respondable = is_respondable

    def set_dynamic(self, is_dynamic):
        self.respondable.set_dynamic(is_dynamic)
        self._is_dynamic = is_dynamic

    def set_position(self, position, relative_to=None):
        self.respondable.set_position(position, relative_to)
    def get_position(self, relative_to=None):
        return self.respondable.get_position(relative_to)
    
    def set_orientation(self, orientation, relative_to=None):
        self.respondable.set_orientation(orientation, relative_to)
    def get_orientation(self, relative_to=None):
        return self.respondable.get_orientation(relative_to)
    
    def set_pose(self, pose, relative_to=None):
            self.respondable.set_pose(pose, relative_to)
    def get_pose(self, relative_to=None):
            return self.respondable.get_pose(relative_to)

    def set_name(self, name):
        self.visible.set_name("{}_visible".format(name))
        self.respondable.set_name("{}_respondable".format(name))

    def get_name(self):
        return self.respondable.get_name(), self.visible.get_name()

    def get_handle(self):
        return self.respondable.get_handle()

    def check_distance(self, object):
        return self.respondable.check_distance(object)

    def remove(self):
        self.visible.remove()
        self.respondable.remove()

    def save_model(self, save_path):
        self.respondable.set_model(True)
        self.respondable.save_model(save_path)

    def set_parent(self, parent_object):
        self.respondable.set_parent(parent_object)
    
    @staticmethod
    def set_emission_color(object, color):
        """set object emission color

        Args:
                object (Shape): [PyRep Shape class]
                color (list): [3 value of rgb] 0 ~ 1
        """
        sim.simSetShapeColor(
        object.get_handle(), None, sim.sim_colorcomponent_emission, color)
    
    @staticmethod
    def set_transparency(object, value):
        """set object transparency

        Args:
                object (Shape): [PyRep Shape class]
                value (list): [list of 1 value] 0 ~ 1
        """
        sim.simSetShapeColor(
        object.get_handle(), None, sim.sim_colorcomponent_transparency, value)

def convex_decompose(obj):
    return obj.get_convex_decomposition(morph=False, vhacd_res=1000000,
                                                                            use_vhacd=True)

@timeout(6000)
def convert_mesh_to_scene_object(env, mesh_file, save_path, bbox_size, target_name=None):
    if target_name is None:
        target_name = get_file_name(mesh_file)
    
    # print("Save {} \n\t===> {}".format(target_name, save_path))

    #1. object base
    base = Dummy.create()
    
    #2. Dummy visible part for scaling 
    visible = Shape.import_shape(filename=mesh_file,
                                                             scaling_factor=1)
    
    #3. Real visible part
    obj_bbox = visible.get_bounding_box()
    obj_bbox_size = [obj_bbox[2*i+1]-obj_bbox[2*i] for i in range(3)]    
    min_idx, min_value = np.argmin(obj_bbox_size), np.min(obj_bbox_size)
    max_idx, max_value = np.argmax(obj_bbox_size), np.max(obj_bbox_size)
    
    scaling_factor = bbox_size/max_value
    visible.remove()
    visible = Shape.import_shape(filename=mesh_file,
                                                            scaling_factor=scaling_factor)
    
    #4. Respondable part
    try:
        respondable = convex_decompose(visible)
        respondable.compute_mass_and_inertia(500)
    except:
        print("Fail to convexdecompose {}".format(target_name))
        return False, scaling_factor
    
    #5. Set property and name
    base.set_name("{}_base".format(target_name))
    visible.set_name("{}_visible".format(target_name))
    respondable.set_name("{}_respondable".format(target_name))
    
    SceneObject.set_transparency(respondable, [0.01])
    
    visible.set_parent(respondable)
    base.set_parent(respondable)
    
    respondable.set_model(True)
    respondable.save_model(save_path)
    
    respondable.remove()

    env.step()
    
    return True, scaling_factor

def visualize_scene_objects(model_list):
    """Load Scene Objects from saved model files
    """
    print("Show {} Scene Models".format(len(model_list)))
    
    # set whole grid
    grid_size = math.sqrt(len(model_list))
    if grid_size.is_integer():
        grid_size = int(grid_size)
    else:
        grid_size = int(grid_size) + 1

    grid = range(grid_size)
    grid_x, grid_y = np.meshgrid(grid, grid)
    object_xy_idx = zip(grid_x.flatten(), grid_y.flatten())

    env = BaseEnv(headless=True)
    start_xy = (-2.25, -2.25)
    step_size = 5 / grid_size
    for model_path, xy_idx in zip(model_list, object_xy_idx):
        obj = env.load_scene_object_from_file(model_path)
        obj.set_dynamic(False)
        obj.set_renderable(False)

        obj_name = os.path.splitext(model_path)[0].split('/')[-1]
        print("Load {}".format(obj_name))

        obj_pos = obj.get_position()
        obj_pos[0] = start_xy[0] + xy_idx[0]*step_size
        obj_pos[1] = start_xy[1] + xy_idx[1]*step_size
        obj.set_position(obj_pos)
        
        env.step()
    for i in range(300):
        env.step()
    env.pr.export_scene("test_after.ttt")
    env.stop()

def visualize_mesh_objects(mesh_list, scaling_factor, env=None):
    """Load Scene Objects from saved model files
    """
    print("Show {} Mesh files".format(len(mesh_list)))
    
    # set whole grid
    grid_size = math.sqrt(len(mesh_list))
    if grid_size.is_integer():
        grid_size = int(grid_size)
    else:
        grid_size = int(grid_size) + 1

    grid = range(grid_size)
    grid_x, grid_y = np.meshgrid(grid, grid)
    object_xy_idx = zip(grid_x.flatten(), grid_y.flatten())

    if env is None:
        env = BaseEnv()
    start_xy = (-2.25, -2.25)
    step_size = 5 / grid_size
    obj_list = []
    for mesh_path, xy_idx in zip(mesh_list, object_xy_idx):
        obj = env.load_mesh_from_file(mesh_path, scaling_factor=scaling_factor)
        obj_list.append(obj)
        obj_pos = obj.get_position()
        obj_pos[0] = start_xy[0] + xy_idx[0]*step_size
        obj_pos[1] = start_xy[1] + xy_idx[1]*step_size
        obj.set_position(obj_pos)
        
        env.step()

    try:
        while True:
            env.step()
    except KeyboardInterrupt:
        for obj in obj_list:
            obj.remove()
        env.step()
        return None
    
    

if __name__=="__main__":
    model_list = [os.path.join(p, "model.ttm") for p in get_dir_list("/data/dataset/uop/UOPData/YCB")]
    visualize_scene_objects(model_list)
    
    # whole_model_list = []
    # object_scaling_factors = {}
    # for object_dir in get_dir_list("3DNet_scene_model"):
    #     model_list = get_file_list(object_dir)
    # visualize_scene_objects(model_list)
        