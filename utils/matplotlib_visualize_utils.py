
import gc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d



class Arrow3D(FancyArrowPatch):
  def __init__(self, xs, ys, zs, *args, **kwargs):
    FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
    self._verts3d = xs, ys, zs

  def draw(self, renderer):
    xs3d, ys3d, zs3d = self._verts3d
    xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
    self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
    FancyArrowPatch.draw(self, renderer)

  def do_3d_projection(self, renderer=None):
    xs3d, ys3d, zs3d = self._verts3d
    xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
    self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

    return np.min(zs)


def visualize_data(data):
    cols = 2
    rows = 1
    fig = plt.figure(figsize=(5*cols, 5*rows), constrained_layout=True)
    fig_idx = 1
    
    input_points = data['points']
    pred_label = np.argmax(data['ins_labels'], axis=1)
    
    label2color = get_label2color(pred_label)
    
    ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
    visualize_input_points(ax, input_points)
    fig_idx += 1
    
    
    ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
    visualize_pred_points(ax, input_points, pred_label, label2color)
    fig_idx += 1

    plt.show()
    

def visualize_uop_result(exp_result):
    # initialize figure
    cols = 4
    rows = 2
    fig = plt.figure(figsize=(5*cols, 5*rows), constrained_layout=True)
    fig_idx = 1
    
    # inference result
    input_points = exp_result['input']
    pred_label = exp_result['pred']
    plane_info = exp_result['plane']
    rot_mat = exp_result['rot']
    
    label2color = get_label2color(pred_label)
    
    ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
    visualize_input_points(ax, input_points)
    fig_idx += 1
    
    ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
    visualize_pred_points(ax, input_points, pred_label, label2color)
    fig_idx += 1
    
    if not plane_info == {}:
        ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
        visualize_plane(ax, input_points, pred_label, plane_info, label2color)
        fig_idx += 1

        ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
        visualize_pred_points(ax, input_points, pred_label, label2color)
        visualize_normal(ax, rot_mat)
        fig_idx += 1
    
    
def visualize_trimesh_result(exp_result):
    # initialize figure
    cols = 3
    rows = 2
    fig = plt.figure(figsize=(5*cols, 5*rows), constrained_layout=True)
    fig_idx = 1
    
    # inference result
    input_points = exp_result['input']
    triangles = exp_result['triangles']
    vertices = exp_result['vertices']
    rot_mat = exp_result['rot']
    
    # input points
    ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
    visualize_input_points(ax, input_points)
    fig_idx += 1
    
    # input points and mesh
    ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
    visualize_input_points(ax, input_points)
    visualize_mesh(ax, triangles, vertices)
    fig_idx += 1
    
    # input points and mesh and arrow
    ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
    visualize_input_points(ax, input_points)
    visualize_mesh(ax, triangles, vertices)
    visualize_normal(ax, rot_mat)
    fig_idx += 1
    
    plt.show()
    

def visualize_ransac_result(exp_result):
    # initialize figure
    cols = 4
    rows = 2
    fig = plt.figure(figsize=(5*cols, 5*rows), constrained_layout=True)
    fig_idx = 1
    
    # inference result
    input_points = exp_result['input']
    pred_points = exp_result['pred_points']
    pred_label = exp_result['pred']
    plane_info = exp_result['plane']
    rot_mat = exp_result['rot']
    
    label2color = get_label2color(pred_label)
    
    ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
    visualize_input_points(ax, input_points)
    fig_idx += 1
    
    ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
    visualize_pred_points(ax, pred_points, pred_label, label2color)
    fig_idx += 1
    
    if not plane_info == {}:
        ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
        visualize_plane(ax, pred_points, pred_label, plane_info, label2color)
        fig_idx += 1

        target_ins = plane_info['instance_idx']
        pred_label[pred_label!=target_ins] = 0

        ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
        visualize_pred_points(ax, pred_points, pred_label, label2color)
        visualize_normal(ax, rot_mat)
        fig_idx += 1
        
        ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
        visualize_pred_points(ax, pred_points, pred_label, label2color)
        fig_idx += 1
        
 
def visualize_bbox_result(exp_result):
    # initialize figure
    cols = 3
    rows = 2
    fig = plt.figure(figsize=(5*cols, 5*rows), constrained_layout=True)
    fig_idx = 1
    
    # inference result
    input_points = exp_result['input']
    bbox_points = exp_result['bbox_points']
    rot_mat = exp_result['rot']
    
    # input points
    ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
    visualize_input_points(ax, input_points)
    fig_idx += 1
    
    # bbox 
    ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
    visualize_input_points(ax, input_points)
    visualize_bbox(ax, bbox_points)
    fig_idx += 1
    
    # input points and bbox and arrow
    ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
    visualize_input_points(ax, input_points)
    visualize_bbox(ax, bbox_points)
    visualize_normal(ax, rot_mat)
    fig_idx += 1
    
    plt.show()
    

def visualize_module_compare(exp_result_dict, save_path=None):
    cols = 3
    rows = 2
    fig = plt.figure(figsize=(5*cols, 5*rows), constrained_layout=True)
    fig_idx = 1
    for mode, eval_result in exp_result_dict.items():
        if mode == 'gt':
            input_points = eval_result['points']
            pred_label = np.argmax(eval_result['ins_labels'], axis=1)
            
            label2color = get_label2color(pred_label)
            
            ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
            visualize_input_points(ax, input_points)
            fig_idx += 1
            
            
            ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
            visualize_pred_points(ax, input_points, pred_label, label2color)
            fig_idx += 1
        elif mode == 'input':
            input_points = eval_result['points']
            ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
            visualize_input_points(ax, input_points)
            ax.set_title(mode)
            fig_idx += 1
            
        elif mode == 'trimesh':
            input_points = eval_result['input']
            triangles = eval_result['triangles']
            vertices = eval_result['vertices']
            rot_mat = eval_result['rot']
            # input points and mesh and arrow
            ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
            visualize_input_points(ax, input_points)
            visualize_mesh(ax, triangles, vertices)
            visualize_normal(ax, rot_mat)
            ax.set_title(mode)
            fig_idx += 1
        elif mode == 'primitive':
            input_points = eval_result['input']
            bbox_points = eval_result['bbox_points']
            rot_mat = eval_result['rot']
            # input points and bbox and arrow
            ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
            visualize_input_points(ax, input_points)
            visualize_bbox(ax, bbox_points)
            visualize_normal(ax, rot_mat)
            ax.set_title(mode)
            fig_idx += 1
        elif mode == 'ransac':
            # inference result
            input_points = eval_result['input']
            pred_points = eval_result['pred_points']
            pred_label = eval_result['pred']
            plane_info = eval_result['plane']
            rot_mat = eval_result['rot']
            label2color = get_label2color(pred_label)
            if not plane_info == {}:
                target_ins = plane_info['instance_idx']
                for ins, color in label2color.items():
                    if ins != target_ins:
                        color[3] = 0.05
                ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
                visualize_pred_points(ax, pred_points, pred_label, label2color)
                visualize_normal(ax, rot_mat)
            ax.set_title(mode)
            fig_idx += 1
        elif 'uop' in mode:
            # inference result
            input_points = eval_result['input']
            pred_label = eval_result['pred']
            plane_info = eval_result['plane']
            rot_mat = eval_result['rot']

            label2color = get_label2color(pred_label)
            
            ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
            ax.set_title(mode)
            if not plane_info == {}:
                target_ins = plane_info['instance_idx']
                for ins, color in label2color.items():
                    if ins != target_ins:
                        color[3] = 0.05
                # ax = fig.add_subplot(rows, cols, fig_idx, projection='3d')
                visualize_pred_points(ax, input_points, pred_label, label2color)
                visualize_normal(ax, rot_mat)
                
            fig_idx += 1
        
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close('all')
        plt.close()
        gc.collect()
    
def visualize_exp_result(exp_result, module='uop', save_path=None):
    visualizer[module](exp_result)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close('all')
        plt.close()
        gc.collect()


visualizer = {
    'gt': visualize_data,
    'uop': visualize_uop_result,
    'trimesh': visualize_trimesh_result,
    'ransac': visualize_ransac_result,
    'primitive': visualize_bbox_result
}


def visualize_input_points(ax, points):
    input_colors = np.zeros((points.shape[0], 4))
    input_colors[:, 3] = 0.1
    draw_info = {
        'xyz': points,
        'title': 'input points',
        'range': 2,
        'c': input_colors, 
    }
    plot3d(ax, **draw_info)
    
def visualize_pred_points(ax, points, pred_label, label2color, title='pred points'):
    pred_colors = np.zeros((points.shape[0], 4))
    
    unique_label = np.unique(pred_label)
    for ins_label in unique_label:
        pred_colors[pred_label==ins_label] = label2color[int(ins_label)]

    draw_info = {
        'xyz': points,
        'title': title,
        'range': 2,
        'c': pred_colors, 
    }
    plot3d(ax, **draw_info)

def visualize_rotated_points(ax, points, rot_mat):
    rotated_points = np.matmul(rot_mat, points.T).T
    input_colors = np.zeros((points.shape[0], 4))
    input_colors[:, 3] = 0.5
    draw_info = {
        'xyz': rotated_points,
        'title': 'rotated points',
        'range': 2,
        'c': input_colors, 
    }
    plot3d(ax, **draw_info)

def visualize_plane(ax, points, pred_label, plane_info, label2color):
    plane_instance = plane_info['instance_idx']
    plane_model = plane_info['plane_eq']
    plane_center = plane_info['plane_center']
    axis_range = 2
    
    pred_colors = np.zeros((points.shape[0], 4))
    pred_colors[:, 3] = 0.01
    pred_colors[pred_label==plane_instance] = label2color[int(plane_instance)]
    
    # target_points = points[pred_label==plane_instance]
    plane_normal = plane_model[:3]
    plane_xyz = [None, None, None]
    eq_idx = np.argmax(abs(plane_normal))
    
    range_idx = list(set([0,1,2]) - set([eq_idx]))
    lin_range = []
    for r_idx in range_idx:
        target = points[:, r_idx]
        lin_range.append([min(target), max(target)])
    
    lin_grid = np.meshgrid(np.linspace(*lin_range[0], 2), np.linspace(*lin_range[1], 2))
    for idx, r_idx in enumerate(range_idx):
        plane_xyz[r_idx] = lin_grid[idx]
    
    plane_xyz[eq_idx] = 0
    for r_idx in range_idx:
        plane_xyz[eq_idx] -= plane_model[r_idx] * plane_xyz[r_idx] 
    plane_xyz[eq_idx] -= plane_model[3]
    plane_xyz[eq_idx] /= plane_model[eq_idx]
    
    plane_xyz = np.array(plane_xyz)
    
    
    draw_info = {
        'xyz': points,
        'title': 'pred plane',
        'range': axis_range,
        'c': pred_colors,
        'plane': plane_xyz
    }
    plot3d(ax, **draw_info)

def visualize_mesh(ax, triangles, vertices):
    vertices = vertices.T
    
    ax.plot_trisurf(vertices[0], vertices[1], vertices[2], triangles=triangles, 
                    color='white', alpha=0.7, linewidth=0.4, edgecolors='gray')

    center = [0, 0, 0]
    ax.set_xlim((center[0]-2/2,center[0]+2/2))
    ax.set_ylim((center[1]-2/2,center[1]+2/2))
    ax.set_zlim((center[2]-2/2,center[2]+2/2))
    ax.set_title('Mesh')
    ax.axis('off')

def visualize_bbox(ax, bbox):
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 6), (1, 7), (2, 5),
        (2, 7), (3, 5), (3, 6), 
        (4, 5), (4, 6), (4, 7)
    ]
    ax.scatter3D(bbox[:, 0], bbox[:, 1], bbox[:, 2], color='g')
    for edge in edges:
        points = np.array(
            [bbox[edge[0]], bbox[edge[1]]]
        )
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color='g')
    
def visualize_normal(ax, rot_mat):
    normal = np.dot(rot_mat.T, np.array([0, 0, -1]).T).T
    xyz = np.array([np.zeros(3), np.zeros(3)+normal]).T
    plot_arrow(ax, xyz, '')

def get_label2color(pred_label):
    # unique label color
    unique_label = np.unique(pred_label)
    label2color = {}
    # remove zero
    if unique_label[0] == 0:
        unique_label = unique_label[1:]
    if len(unique_label) == 0:
        pass
    else:    
        min_instance = unique_label[0]
        max_instance = max(unique_label)
        label_color = plt.get_cmap("tab20")(unique_label / (max_instance if max_instance > 0 else 1))
        label_color[:, 3] = 0.7
        for ins_label in unique_label:
            label2color[int(ins_label)] = label_color[int(ins_label-min_instance)]
    label2color[0] = [0, 0, 0, 0.01]
    
    return label2color

def plot_coordinate(ax):
  arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
  
  a = Arrow3D([0, 1], [0, 0], [0, 0], **arrow_prop_dict, color='r')
  ax.add_artist(a)
  a = Arrow3D([0, 0], [0, 1], [0, 0], **arrow_prop_dict, color='b')
  ax.add_artist(a)
  a = Arrow3D([0, 0], [0, 0], [0, 1], **arrow_prop_dict, color='g')
  ax.add_artist(a)

  # Give them a name:
  # ax.text(0.0, 0.0, -0.1, r'$0$')
  ax.text(1.1, 0, 0, r'$x$')
  ax.text(0, 1.1, 0, r'$y$')
  ax.text(0, 0, 1.1, r'$z$')
  
def plot_arrow(ax, xyz, text):
  arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
  a = Arrow3D(xyz[0], xyz[1], xyz[2], **arrow_prop_dict, color='r')
  ax.add_artist(a)
  ax.text(1.1, 0, 0, text)
  
def plot3d(ax, xyz, center=None, title=None, range=None, c=None, plane=None, axis=False):
    fontlabel = {"fontsize":"large", "color":"gray", "fontweight":"bold"}
    if range is not None:
        if center is None:
            center = [0,0,0] #FIXME
            
        ax.set_xlim((center[0]-range/2,center[0]+range/2))
        ax.set_ylim((center[1]-range/2,center[1]+range/2))
        ax.set_zlim((center[2]-range/2,center[2]+range/2))
    ax.set_title(title, fontdict=fontlabel)
    if plane is not None:
        ax.plot_surface(plane[0], plane[1], plane[2], alpha=0.9)
    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2], depthshade=0, c=c)
    if not axis:
        ax.axis('off')