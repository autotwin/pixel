import argparse
from atpixel import MRI_to_stl as mts
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os 
from pathlib import Path 
from stl import mesh

def create_folder(path: Path):
    """Given a path, will create the folder needed."""
    if not path.is_dir():
        os.makedirs(path)
    return 

def create_skull_still(stl_path_file: Path, vis_path: Path, elev: float=-90, azim: float=90):
    """"""
    fig = plt.figure(figsize=(100,100))
    ax = fig.add_subplot(111, projection="3d", elev=elev, azim=azim)
    imported_mesh = mesh.Mesh.from_file(str(stl_path_file))
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(imported_mesh.vectors,alpha=1.0,ec='k',fc=(0.75,0.75,0.75))) #(0.75,0.75,0.75)
    scale = imported_mesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    plt.axis('off')
    fig.tight_layout()
    plt.savefig(str(vis_path) + '/skull_still_image_elev%i_azim%i.png'%(elev,azim))
    plt.close()
    return

def get_visualization_relevant_path_names(input_file_str: str) -> Path:
    """Given an input file string. Will return visualization folder and stl file paths."""
    input_file = mts.string_to_path(input_file_str)
    input_dict = mts._yml_to_dict(yml_path_file=input_file)
    vis_path_str = input_dict["visualization_folder_name"]
    stl_path_file_outer_str = input_dict["stl_path_file_outer"]
    vis_path = mts.string_to_path(vis_path_str)
    stl_path_file_outer = mts.string_to_path(stl_path_file_outer_str)
    return vis_path, stl_path_file_outer

def run_visualization_code(input_file_str: str) -> None:
    """Given yaml file input string. Run all visualization code developed to date."""
    vis_path, stl_path_file_outer = get_visualization_relevant_path_names(input_file_str)
    create_folder(vis_path)
    create_skull_still(stl_path_file_outer,vis_path)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="the .yml user input file")
    args = parser.parse_args()
    input_file_str = args.input_file
    run_visualization_code(input_file_str)