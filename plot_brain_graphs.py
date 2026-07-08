'''
To run this script:

1) To see everything together (colored by atlas region):
    python3 plot_brain_graphs.py --folder Results/IXI002-Guys-0828-MRA

2) To see a specific region (e.g., Region 12):
    python3 plot_brain_graphs.py --folder Results/IXI002-Guys-0828-MRA --region 12

3) To see the vessels inside the original MRA "ghost" (Transparency):
    python3 plot_brain_graphs.py --folder Results/IXI002-Guys-0828-MRA --mra IXI/MRA/IXI002-Guys-0828-MRA.nii.gz

4) To see a specific VTP file directly:
    python3 plot_brain_graphs.py --vtp Results/IXI002-Guys-0828-MRA/vessel_graph.vtp

5) To see a direct VTP inside the original MRA "ghost":
    python3 plot_brain_graphs.py --vtp Results/IXI002-Guys-0828-MRA/vessel_graph.vtp --mra IXI/MRA/IXI002-Guys-0828-MRA.nii.gz
'''

import os
import vedo
import argparse
import numpy as np
import nibabel as nib

# The flip matrix from the original paper/code
SLICER_MATRIX = np.array([
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])


def visualize_vessels(case_folder=None, vtp_path=None, region_id=None, show_volume=False, mra_path=None):
    plt = vedo.Plotter(title="Vessel Visualization", bg="blackboard")
    actors = []

    # 1. Handle Volume Overlay (Alignment Logic)
    if show_volume and mra_path and os.path.exists(mra_path):
        print(f"Loading and Aligning Volume: {mra_path}")
        nii = nib.load(mra_path)
        vol = vedo.Volume(nii.get_fdata())
        vol.apply_transform(nii.affine.tolist())     # Step A: World Space
        vol.apply_transform(SLICER_MATRIX.tolist())  # Step B: Slicer Flip
        vol.mode(0).alpha([0, 0.1, 0.2])
        actors.append(vol)

    # 2. Case A: User provided a direct path to a VTP file
    if vtp_path:
        if os.path.exists(vtp_path):
            print(f"Loading Direct VTP: {vtp_path}")
            # We use green for direct VTPs to distinguish them from regional ones
            vessels = vedo.load(vtp_path).c("green").lw(2)
            actors.append(vessels)
        else:
            print(f"Error: VTP file not found at {vtp_path}")
            return

    # 3. Case B: User provided a results folder (Regional Mode)
    elif case_folder:
        if region_id is not None:
            # Plot a single region from the folder
            vtp_dir = os.path.join(case_folder, "regional_vtps")
            target_file = [f for f in os.listdir(vtp_dir) if f.startswith(f"region_{region_id}_")]
            
            if target_file:
                path = os.path.join(vtp_dir, target_file[0])
                print(f"Loading Region {region_id}: {path}")
                actors.append(vedo.load(path).c("red").lw(4))
            else:
                print(f"Error: Region {region_id} not found in {vtp_dir}")
                return
        else:
            # Plot the combined regional graph
            combined_vtp = os.path.join(case_folder, "vessel_graph_regional.vtp")
            if os.path.exists(combined_vtp):
                print(f"Loading Combined Regional Graph: {combined_vtp}")
                vessels = vedo.load(combined_vtp)
                vessels.cmap("Paired", input_array="RegionID", on="cells") 
                actors.append(vessels)
                actors.append(vedo.ScalarBar(vessels, title="Region ID"))
            else:
                # Fallback to the basic graph inside the folder
                base_vtp = os.path.join(case_folder, "vessel_graph.vtp")
                if os.path.exists(base_vtp):
                    print("Loading base graph (vessel_graph.vtp)...")
                    actors.append(vedo.load(base_vtp).c("green").lw(2))
    
    else:
        print("Error: You must provide either --folder or --vtp")
        return

    plt.show(actors, axes=1, interactive=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help="Path to the patient's result folder")
    parser.add_argument('--vtp', help="Direct path to a specific .vtp file")
    parser.add_argument('--region', type=int, default=None, help="Specific Region ID to plot (requires --folder)")
    parser.add_argument('--mra', help="Optional path to original MRA .nii.gz for overlay")
    args = parser.parse_args()

    visualize_vessels(
        case_folder=args.folder,
        vtp_path=args.vtp,
        region_id=args.region, 
        show_volume=True if args.mra else False, 
        mra_path=args.mra
    )