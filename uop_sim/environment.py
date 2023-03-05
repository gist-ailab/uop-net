from pyrep.objects.dummy import Dummy
from pyrep import PyRep
import os
import time

from pyrep.objects.shape import Shape
from pyrep.backend import sim
from pyrep.objects.vision_sensor import VisionSensor

from os.path import join
import numpy as np

TABLE_MODEL_PATH = join(os.path.dirname(__file__), 'coppeliasim', "table.ttm")
TILTABLE_TABLE_MODEL_PATH = join(os.path.dirname(__file__), 'coppeliasim', "tiltable_table.ttm")
PLACEMENT_SCENE_FILE = join(os.path.dirname(__file__), 'coppeliasim', "placement_scene.ttt")
INSPECTION_SCENE_FILE = join(os.path.dirname(__file__), 'coppeliasim', "inspection_scene.ttt")
EVALUATE_SCENE_FILE = join(os.path.dirname(__file__), 'coppeliasim', "evaluate_scene.ttt")


class EXPSTATE:
    START = 0
    ING = 1
    END = 2



class SceneObject(object):
    def __init__(self, respondable_part: Shape):
        self.respondable = respondable_part
        self.respondable.compute_mass_and_inertia(500)
        name = self.respondable.get_name()
        if name[-1].isnumeric():
            self.index = int(name[-1]) + 2
            self.name = name[:-1].replace("_respondable", "")
        else:
            self.index = 1
            self.name = name.replace("_respondable", "")

        self.visible = Shape(name.replace("respondable","visible"))
        self.base = Dummy(name.replace("respondable","base"))
        self.visible.set_parent(self.respondable)
        self.base.set_parent(self.respondable)

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

        self.set_activate(False)        

    def set_activate(self, is_activate):
        self.visible.set_detectable(is_activate)
        self.visible.set_renderable(is_activate)
        
        self.respondable.set_collidable(is_activate)
        self.respondable.set_measurable(is_activate)
        self.respondable.set_dynamic(is_activate)
        self.respondable.set_respondable(is_activate)
        
        self._is_activate = is_activate
        
    def initialize_visible(self):
        self.visible.set_collidable(False)
        self.visible.set_measurable(False)
        self.visible.set_detectable(True)
        self.visible.set_renderable(True)
        self.visible.set_dynamic(False)
        self.visible.set_respondable(False)
        self.set_emission_color(self.visible, [1, 1, 1])

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
        if is_dynamic:
            if not self._is_respondable:
                self.set_respondable(True)
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

    def set_matrix(self, matrix, relative_to=None):
        self.respondable.set_matrix(matrix, relative_to)
    def get_matrix(self, relative_to=None):
        return self.respondable.get_matrix(relative_to)

    def set_name(self, name):
        self.visible.set_name("{}_visible".format(name))
        self.respondable.set_name("{}_respondable".format(name))
        self.name = name

    def get_name(self):
        return self.name

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



class Table(SceneObject):
    def __init__(self, model_base):
        table = model_base
        
        table_top = None
        table_zaxis = None
        visible = None
        camera_base = None

        for child in model_base.get_objects_in_tree(exclude_base=True):
            if "top" in child.get_name():
                table_top = child
            elif "zaxis" in child.get_name():
                table_zaxis = child
            elif "visible" in child.get_name():
                visible = child
            elif "camera_base" in child.get_name():
                camera_base = child
            
        
        assert table_top, 'There is no table top'
        assert table_zaxis, 'There is no table zaxis'
        assert visible, 'There is no visible part'
        assert camera_base, 'There is no camera base'
        
        self.respondable = table
        self.visible = visible
        
        self.table_top = table_top
        self.table_zaxis = table_zaxis
        self.camera_base = camera_base

        exp_bbox = self.respondable.get_bounding_box() # min x, max x, min y, max y, min z, max z
        self.xy_offset = (1, 1)
        # self.xz_offset = (3, 3)
    
    def set_table_position(self, exp_xy):
        pos = self.respondable.get_position()
        
        pos[0] = self.xy_offset[0] * exp_xy[0]
        pos[1] = self.xy_offset[1] * exp_xy[1]
        
        self.respondable.set_position(pos)

    def get_zaxis(self, relative_to=None):
        world = self.table_top.get_position(relative_to=relative_to)
        zpoint = self.table_zaxis.get_position(relative_to=relative_to)

        return np.array(zpoint) - np.array(world)



class PlacementExp():
    def __init__(self, tolerance, max_step, 
                             table, exp_idx):
        self.tolerance = tolerance
        self.max_step = max_step

        self.table = table
        self.table.set_table_position(exp_idx)
        self.target_object = None
        
        self.movements = []
        
        self.state = EXPSTATE.START
        self.stability = 0
        self.step_count = 0
        self.last_5_movement = []
        self.start_z = None
        self.end_z = None
        
    def step_calculate(self):
        assert self.target_object is not None
        '''calculate movement with transform matrix'''
        current_mat = self.target_object.get_matrix(relative_to=self.table.table_top)
        current_z = self.target_object.get_position()[2]
        previous = np.array(self.previous_mat)
        current = np.array(current_mat)
        movement = np.linalg.norm(previous - current)
    
        '''update'''
        self.previous_mat = current_mat
        self.stability += movement
        self.step_count += 1
        self.last_5_movement.append(movement)
        
        if len(self.last_5_movement) > 5:
            self.last_5_movement.pop(0)
            if np.sum(self.last_5_movement) < self.tolerance:
                self.state = EXPSTATE.END
                self.target_object.set_activate(False)
            else:
                if self.step_count > self.max_step:
                    self.stability += 100
                    self.state = EXPSTATE.END
                    self.target_object.set_activate(False)            
                elif current_z < 0.3:
                    self.stability += 100
                    self.state = EXPSTATE.END
                    self.target_object.set_activate(False)

    def init(self, orientation, target_object):
        self.target_object = target_object

        self.target_object.set_activate(False)
        self.target_object.set_position([0, 0, 0.4], relative_to=self.table.table_top)
        self.target_object.set_orientation(orientation)
        
        self.start_z = self.table.get_zaxis(relative_to=self.target_object.base)
        
        # check distance and move to table
        z_distance = self.target_object.check_distance(self.table.respondable)
        self.target_object.set_position([0, 0, 0.4 - z_distance], relative_to=self.table.table_top)
        
        self.previous_mat = self.target_object.get_matrix()
        
        self.target_object.set_activate(True)
        self.state = EXPSTATE.ING
    
    def calculate_z_axis(self):
        self.target_object.base
        
    def reset(self):
        stability = self.stability
        last_mat = self.previous_mat
        start_z = self.start_z
        end_z = self.table.get_zaxis(relative_to=self.target_object.base)
        
        self.stability = 0
        self.step_count = 0
        self.last_5_movement = []
        self.state = EXPSTATE.START
        
        return stability, last_mat, start_z, end_z



class PlacementEnv():
    
    def __init__(self, grid_size, headless=False, time_step=0.005,
                             tolerance=1e-5, max_step=1000):
        
        self.pr = PyRep()
        # Launch the application with a scene file in headless mode
        self.pr.launch(PLACEMENT_SCENE_FILE, headless=headless) 
        self.pr.set_simulation_timestep(time_step) # 0.01 = 10ms
        self.pr.start()    # Start the simulation

        self._initialize_env(grid_size, tolerance, max_step)
        
        self.tolerance = tolerance
        self.max_step = max_step

        for _ in range(10):
            self.pr.step()
    
    def _initialize_env(self, grid_size, tolerance, max_step):
        exp_per_iter = grid_size**2
        grid_x, grid_y = np.meshgrid(range(grid_size), range(grid_size))
        
        self.experiments = []
        
        for exp_id in range(exp_per_iter):
            exp_xy = [grid_x.flatten()[exp_id], grid_y.flatten()[exp_id]]
            
            table = Table(self.import_model(TABLE_MODEL_PATH))
            exp = PlacementExp(tolerance, max_step, table, exp_xy)
            self.experiments.append(exp)
    
    def get_stability(self, model_path, orientation_list):
        stability = {orientation: None for orientation in orientation_list}
        self.target_objects = [self.load_scene_object_from_file(model_path) for _ in range(len(self.experiments))]
    
        self.exp_ori_list = list(range(len(self.experiments)))
        is_finished = False
        next_ori_idx = 0
        while not is_finished:
            self.pr.step()
            is_finished = True
            for exp_id, exp in enumerate(self.experiments):
                if exp.state==EXPSTATE.START:
                    if next_ori_idx >= len(orientation_list):
                        continue
                    exp.init(orientation_list[next_ori_idx], self.target_objects[exp_id])
                    self.exp_ori_list[exp_id] = next_ori_idx
                    next_ori_idx += 1
                    print("Simulate orientation >>> {} / {}".format(next_ori_idx, len(orientation_list)), end='\r')
                    is_finished = False
                elif exp.state==EXPSTATE.ING:
                    is_finished = False
                    exp.step_calculate()
                elif exp.state==EXPSTATE.END:
                    stability[orientation_list[self.exp_ori_list[exp_id]]] = exp.reset()
                    if next_ori_idx < len(orientation_list):
                        is_finished = False
                else:
                    raise NotImplementedError
            
        for obj in self.target_objects:
            obj.remove()
        self.target_objects = []
        
        return stability
    
    def stop(self):
        self.pr.stop()    # Stop the simulation
        self.pr.shutdown()    # Close the application

    def step(self):
        self.pr.step()
    
    def load_scene_object_from_file(self, file_path):
        file_name = os.path.splitext(file_path)[0].split("/")[-1]
        respondable = self.import_model(file_path)
        return SceneObject(respondable_part=respondable)

    def import_model(self, model_path):
        return self.pr.import_model(model_path)



class TiltableTable(SceneObject):
    def __init__(self, model_base):
        table = model_base
        
        table_top = None
        visible = None

        for child in model_base.get_objects_in_tree(exclude_base=True):
            if "top" in child.get_name():
                table_top = child
            elif "visible" in child.get_name():
                visible = child
        
        assert table_top, 'There is no table top'
        assert visible, 'There is no visible part'
        
        self.respondable = table
        self.visible = visible
        self.table_top = table_top

        exp_bbox = self.respondable.get_bounding_box() # min x, max x, min y, max y, min z, max z
        self.xz_offset = (2, 2)
    
    def initialize_pose(self, position, z_rot):
        self.respondable.set_position(position)
        self.table_top.set_parent(None)
        rotation = [0, 0, z_rot]
        self.respondable.rotate(rotation)
        self.inital_pose = self.respondable.get_pose()

    def reset(self):
        self.respondable.set_pose(self.inital_pose)

    def tilt(self, tilt):
        rotation = [0, tilt, 0]
        self.respondable.rotate(rotation)
        
    def set_target_object(self, target_object, matrix):
        target_object.set_matrix(matrix, relative_to=self.table_top)
        matrix = target_object.get_matrix()
        
        position = target_object.get_position(relative_to=self.respondable)
        position[2] = position[2] + 0.1
        target_object.set_position(position, relative_to=self.respondable)
        
        # check distance and move to table
        position = target_object.get_position(relative_to=self.respondable)
        z_distance = target_object.check_distance(self.respondable)
        position[2] = position[2] - z_distance
        target_object.set_position(position, relative_to=self.respondable)



class InspectionExp():
    def __init__(self, tolerance, min_step, table):
        self.tolerance = tolerance
        self.min_step = min_step
        
        self.table = table
        self.target_object = None
        
        self.is_stable = False
        self.stability = 0
        self.step_count = 0
        self.total_movements = []
        
    def step_calculate(self):
        assert self.target_object is not None
        '''calculate movement with transform matrix'''
        current_mat = self.target_object.get_matrix()
        previous = np.array(self.previous_mat)
        current = np.array(current_mat)
        movement = np.linalg.norm(previous - current)
        current_z = self.target_object.get_position()[2]
        
        '''update'''
        self.previous_mat = current_mat
        self.stability += np.linalg.norm(self.initial_mat - current_mat)
        self.step_count += 1
        self.total_movements.append(movement)
        
        if self.step_count == self.min_step:
            self.state = EXPSTATE.END
            if self.stability < self.tolerance:
                self.is_stable = True
                self.target_object.set_activate(False)
            else:
                self.is_stable = False
                self.target_object.set_activate(False)            
        else:
            if current_z < 0.3:
                self.state = EXPSTATE.END
                self.is_stable = False
                self.target_object.set_activate(False)
            if self.stability > self.tolerance:
                self.state = EXPSTATE.END
                self.is_stable = False
                self.target_object.set_activate(False)
                
    def init(self, target_object, matrix):
        self.target_object = target_object
        self.table.reset()
        self.target_object.set_activate(False)
        self.table.set_target_object(self.target_object, matrix)
    
    def init_mat(self):
        self.previous_mat = self.target_object.get_matrix()
        self.initial_mat = self.previous_mat
    
    def start(self):
        self.target_object.set_activate(True)#FIXME
        self.state = EXPSTATE.ING
    
    def reset(self):
        self.target_object.set_activate(False)
        is_stable = self.is_stable
        
        self.stability = 0
        self.is_stable = False
        self.step_count = 0
        self.total_movements = []
        self.state = EXPSTATE.END
        
    def tilt(self, tilt):
        self.table.tilt(tilt)


class InspectionEnv():
    def __init__(self, headless=False, time_step=0.005, tilt=5,
                             tolerance=0.3, min_step=10):
        self.pr = PyRep()

        # Launch the application with a scene file in headless mode
        self.pr.launch(INSPECTION_SCENE_FILE, headless=headless) 
        self.pr.set_simulation_timestep(time_step) # 0.0001 ~ 10
        self.min_step = min_step
        
        self.pr.start()    # Start the simulation

        self.tilt = np.pi *(tilt/180)
        self._initialize_env(tolerance, min_step)
    
    def _initialize_env(self, tolerance, min_step):
        grid_x_pos, grid_y_pos = np.meshgrid(np.linspace(-4, 4, 5), np.linspace(-4, 4, 5))
        grid_z_rot = np.linspace(0, 2*np.pi, 26)[:25]

        self.experiments = []
        for exp_idx in range(25): # 5*5 table
            table = TiltableTable(self.import_model(TILTABLE_TABLE_MODEL_PATH))
            # set table pos to grid
            pos_xy = [grid_x_pos.flatten()[exp_idx], grid_y_pos.flatten()[exp_idx]]
            position = pos_xy + [table.get_position()[2]]

            table.initialize_pose(position, grid_z_rot[exp_idx])

            exp = InspectionExp(tolerance, min_step, table)
            self.experiments.append(exp)
        
    def inspect_clustered_info(self, model_path, clustered_info):
        self.target_objects = [self.load_scene_object_from_file(model_path) for _ in range(len(self.experiments))]
        
        inspected_info = {}
        for idx, (orientation, info) in enumerate(clustered_info.items()):
            print("Inspection Clustered Pose >>> {:<2} / {:<2}".format(idx, len(clustered_info.keys())), end='\r')
            matrix = info['matrix']
            
            is_stable = True
            
            for exp_id, exp in enumerate(self.experiments):
                exp.init(self.target_objects[exp_id], matrix)
            
            batch_size = 25
            for batch_idx in range(1):
                s, e = batch_size*(batch_idx), batch_size*(batch_idx+1)
                batch_experiments = self.experiments[s:e]
                for exp in batch_experiments:
                    exp.start()
            
                for step in range(1, 10):
                    self.step()
                
                    # for exp in batch_experiments:
                    #     if exp.state == EXPSTATE.END:
                    #         is_stable = False
                    #         continue
                    #     exp.step_calculate()
                
                for i in range(50):    
                    for exp in batch_experiments:
                        exp.tilt(self.tilt/50)
                    self.step()
                for exp in batch_experiments:
                    exp.init_mat()
            
                for step in range(1, 5000):
                    print("Inspection Clustered Pose >>> {:<2} / {:<2} | {:<3} / {:<3}".format(idx, len(clustered_info.keys()), step, 200), end='\r')
                    ret = self.step()
                    if not ret:
                        is_stable = False
                        break
                
                    for exp in batch_experiments:
                        if exp.state == EXPSTATE.END:
                            is_stable = False
                            break
                        exp.step_calculate()
                    if not is_stable:
                        break
            # select target experiment
            
            #by stacked movements
            # stacked_movements = []
            # for exp in self.experiments:
            #     stacked_movements.append(exp.stability)
            
            # target_exp = self.experiments[np.argmax(stacked_movements)]

            # target_stabilities = {
            #     "total_movements": target_exp.total_movements,
            #     "sum": target_exp.stability,
            #     "is_stable": target_exp.is_stable
            # }
            
            for exp in self.experiments:
                exp.reset()
            
            if is_stable:
                inspected_info[orientation] = clustered_info[orientation]
                # inspected_info[orientation].update(target_stabilities)
            
        for obj in self.target_objects:
            obj.remove()
        
        self.target_objects = []
        
        print("Inspection Clustered Pose >>> {} to {}".format(len(clustered_info.keys()), len(inspected_info.keys())))
        return inspected_info
        
    def stop(self):
        self.pr.stop()    # Stop the simulation
        self.pr.shutdown()    # Close the application

    def step(self):
        t1 = time.time()
        self.pr.step()
        delta = time.time() - t1
        if delta > 10:
            print("Too long to 1 step")
            return False
        else:
            return True
        
        
    
    def load_scene_object_from_file(self, file_path):
        file_name = os.path.splitext(file_path)[0].split("/")[-1]
        respondable = self.import_model(file_path)
        return SceneObject(respondable_part=respondable)

    def import_model(self, model_path):
        return self.pr.import_model(model_path)



class EvaluateEnv():
    
    def __init__(self, headless=False, time_step=0.005, tolerance=1e-5,
                             max_step=1000, demo=False):
        self.pr = PyRep()

        # Launch the application with a scene file in headless mode
        self.pr.launch(EVALUATE_SCENE_FILE, headless=headless) 
        self.pr.set_simulation_timestep(time_step) # 0.01 = 10ms
        self.pr.start()    # Start the simulation
        
        self.tolerance = tolerance
        self.max_step = max_step
        
        self.demo = demo
        
        self.table = Shape('diningTable_visible')
        
        self.object_base = Dummy('object_base')
        self.camera_depth = VisionSensor('camera_depth')
        self.camera_rgb = VisionSensor('camera_rgb')
        self.camera_obs = VisionSensor('camera_obs')
        
        if not self.demo:
            self.camera_depth.set_resolution([1, 1])
            self.camera_rgb.set_resolution([1, 1])
            self.camera_obs.set_resolution([1, 1])
        
        self.step()
        self.target = None


    def reset(self, model_path=None):
        if model_path is not None:
            if self.target is not None:
                self.target.remove()
            self.target = self.load_scene_object_from_file(model_path)
            self.init_ori = self.target.get_orientation()
            
        assert self.target is not None

        self.target.set_position([0, 0, 0], relative_to=self.object_base)
        
        if self.demo:
            rand_ori = np.random.rand(3) * np.pi * 2
            self.target.set_orientation(rand_ori)
        else:
            self.target.set_orientation(self.init_ori)
        self.target.respondable.set_parent(self.object_base)
        
        self.target.initialize_visible()
        
        self.initial_pose = self.target.get_pose()
        
        self.step()

    def initialize(self):
        self.target.set_pose(self.initial_pose)
        self.target.initialize_visible()
        self.step()

    def observe(self, return_rgb=False):
        
        depth = self.camera_depth.capture_depth()
        object_mask = depth < 1
        depth[object_mask] = 0
        
        point_cloud = self.camera_depth.capture_pointcloud()
        
        point_cloud = point_cloud[object_mask]
        
        if return_rgb:
            rgb = self.camera_rgb.capture_rgb()
            return point_cloud, rgb
        
        return point_cloud
    
    def evaluate(self, rot, observe=False):
        mat = self.target.get_matrix()
        mat[:3, :3] = np.dot(rot, mat[:3, :3])
        self.target.set_matrix(mat)
        
        position = self.target.get_position(relative_to=self.table)
        z_distance = self.target.check_distance(self.table)
        position[2] = position[2] - z_distance
        self.target.set_position(position, relative_to=self.table)
        self.target.set_activate(True)
        self.pr.step()

        is_stop = False
        stability = 0.
        previous_mat = self.target.get_matrix()
        step_count = 0
        last_5_movement = []
        movements = []
        matrixs = [previous_mat]
        observations = []
        if observe:
            observations.append(self.camera_obs.capture_rgb())
        while not is_stop:
            self.pr.step()
            current_mat = self.target.get_matrix()
            matrixs.append(current_mat)
            previous = np.array(previous_mat)
        
            current = np.array(current_mat)
            movement = np.linalg.norm(previous - current)
        
            previous_mat = current_mat
            stability += movement
            # if observe:
            #     observations.append(self.observe()[0])

            step_count += 1

            last_5_movement.append(movement)
            movements.append(movement)
            if len(last_5_movement) > 5:
                last_5_movement.pop(0)
            
            if np.mean(last_5_movement) < self.tolerance:
                self.target.set_activate(False)
                is_stop = True
            else:
                if step_count > self.max_step:
                    stability += 100
                    self.target.set_activate(False)
                    is_stop = True
        
        if observe:
            observations.append(self.camera_obs.capture_rgb())
        
        return {
            'stability': stability,
            'step_count': step_count,
            'movements': movements,
            'observations': observations,
            'matrix': matrixs,
        }
    
    def step(self):
        self.pr.step()

    def stop(self):
        self.pr.stop()    # Stop the simulation
        self.pr.shutdown()    # Close the application

    def load_scene_object_from_file(self, file_path):
        file_name = os.path.splitext(file_path)[0].split("/")[-1]
        respondable = self.import_model(file_path)
        return SceneObject(respondable_part=respondable)

    def import_model(self, model_path):
        return self.pr.import_model(model_path)


if __name__=='__main__':
    pr = PyRep()
    
    # Launch the application with a scene file in headless mode
    pr.launch(scene_file=INSPECTION_SCENE_FILE, headless=True) 
    pr.start()    # Start the simulation
    
    grid_x, grid_y = np.meshgrid(range(-4, 5, 2), range(-4, 5, 2))
    
    grid_y_rot, grid_z_rot = np.meshgrid(np.linspace(0, np.pi/18, 6)[:5], np.linspace(0, 2*np.pi, 6)[:5])
    
    for idx in range(25):
        table = pr.import_model(TILTABLE_TABLE_MODEL_PATH)
        pos_xy = [grid_x.flatten()[idx], grid_y.flatten()[idx]]
        position = pos_xy + [table.get_position()[2]]
        table.set_position(position)    
        
        rot_yz = [grid_y_rot.flatten()[idx], grid_z_rot.flatten()[idx]]
        rotation = [0] + rot_yz
        table.rotate(rotation)
    
    