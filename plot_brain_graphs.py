'''
To run this script:

1) To see the new MNI-aligned atlas graph (colored by atlas region):
    python3 plot_brain_graphs.py --folder Results/IXI002-Guys-0828-MRA

2) To see a specific region (e.g., Region 12) from the MNI atlas:
    python3 plot_brain_graphs.py --folder Results/IXI002-Guys-0828-MRA --region 12

3) To see the MNI vessels inside the MNI Template "ghost" (Transparency):
    # Note: Because the new graphs are in MNI space, you must overlay them on the MNI template!
    python3 plot_brain_graphs.py --folder Results/IXI002-Guys-0828-MRA --volume Atlas/Atlas_182_MNI152/MNI152_T1_1mm_Brain.nii.gz

4) To see the OLD NATIVE vessels inside the original native MRA "ghost":
    python3 plot_brain_graphs.py --vtp Results/IXI002-Guys-0828-MRA/vessel_graph.vtp --volume IXI/MRA/IXI002-Guys-0828-MRA.nii.gz
'''
import os
import glob
import vedo
import argparse
import numpy as np
import nibabel as nib

# The flip matrix to map NIfTI (RAS) space to Slicer/Graph (LPS) space
SLICER_MATRIX = np.array([
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

def visualize_vessels(case_folder=None, vtp_path=None, region_id=None, show_volume=False, volume_path=None):
    plt = vedo.Plotter(title="Vessel Atlas Visualization", bg="blackboard")
    actors = []

    # 1. Handle Volume Overlay
    if show_volume and volume_path and os.path.exists(volume_path):
        print(f"Loading and Aligning Volume: {volume_path}")
        nii = nib.load(volume_path)
        vol = vedo.Volume(nii.get_fdata())
        
        # Apply standard NIfTI affine, then flip to LPS space to match the graph coordinates
        vol.apply_transform(nii.affine.tolist())     
        vol.apply_transform(SLICER_MATRIX.tolist())  
        
        # Lower alpha to see vessels better through the tissue
        vol.mode(1).alpha([0, 0.1, 0.15]) 
        actors.append(vol)

    # 2. Case A: User provided a direct path to a VTP file
    if vtp_path:
        if os.path.exists(vtp_path):
            print(f"Loading Direct VTP: {vtp_path}")
            vessels = vedo.load(vtp_path).lw(2)
            # If the VTP has RegionID, color it
            if "RegionID" in vessels.celldata.keys():
                vessels.cmap("Paired", input_array="RegionID", on="cells")
            else:
                vessels.c("green")
            actors.append(vessels)
        else:
            print(f"Error: VTP file not found at {vtp_path}")
            return

    # 3. Case B: User provided a results folder (New Atlas Mode)
    elif case_folder:
        # First, try to find the new MNI-aligned VTP (dynamically named)
        atlas_vtp = None
        mni_vtps = glob.glob(os.path.join(case_folder, "*_vessel_graph_labeled_MNI.vtp"))
        
        if mni_vtps:
            atlas_vtp = mni_vtps[0] # Grab the first one found
        else:
            # Fallbacks for older versions of the pipeline
            atlas_vtp = os.path.join(case_folder, "vessel_graph_labeled_atlas.vtp")
            if not os.path.exists(atlas_vtp):
                atlas_vtp = os.path.join(case_folder, "vessel_graph.vtp")

        if os.path.exists(atlas_vtp):
            print(f"Loading Atlas Graph: {atlas_vtp}")
            vessels = vedo.load(atlas_vtp)
            
            if "RegionID" in vessels.celldata.keys():
                if region_id is not None:
                    print(f"Filtering for Region ID: {region_id}")
                    vessels.threshold("RegionID", region_id, region_id)
                    vessels.c("red").lw(4)
                else:
                    # 1. Get the real Region IDs
                    real_ids = vessels.celldata["RegionID"]
                    
                    # 2. Scramble them in memory for high visual contrast
                    # (rid * 137) % 256 ensures neighboring IDs (1, 2, 3) get very different colors
                    visual_ids = (real_ids.astype(int) * 137) % 256
                    
                    # 3. Add this temporary array to the vessels object
                    vessels.celldata["VisualID"] = visual_ids.astype(np.int32)
                    
                    # 4. Use "gist_ncar" (256 colors) instead of "Paired" (12 colors)
                    vessels.cmap("gist_ncar", input_array="VisualID", on="cells").lw(2)
                    
                    # Add a legend that shows the REAL IDs (not the scrambled ones)
                    actors.append(vedo.ScalarBar(vessels, title="Region (Visual Map)"))
            else:
                print("Warning: No RegionID found in VTP, showing default green.")
                vessels.c("green").lw(2)
            
            actors.append(vessels)
        else:
            print(f"Error: No vessel graph found in {case_folder}")
            return
    
    else:
        print("Error: You must provide either --folder or --vtp")
        return

    plt.show(actors, axes=1, interactive=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help="Path to the patient's result folder")
    parser.add_argument('--vtp', help="Direct path to a specific .vtp file")
    parser.add_argument('--region', type=int, default=None, help="Specific Region ID to extract from the global graph")
    
    # Changed from --mra to --volume, with an alias to preserve backward compatibility
    parser.add_argument('--volume', '--mra', dest='volume', help="Optional path to NIfTI volume (use MNI Template for MNI graphs, original MRA for native graphs) for overlay")
    
    args = parser.parse_args()

    visualize_vessels(
        case_folder=args.folder,
        vtp_path=args.vtp,
        region_id=args.region, 
        show_volume=True if args.volume else False, 
        volume_path=args.volume
    )