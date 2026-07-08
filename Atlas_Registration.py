import logging
import os
import shutil
import numpy as np
import nibabel as nib
from nipype.interfaces import fsl

logger = logging.getLogger(__name__)

def _fsl_available() -> bool:
    required_binaries = ["bet", "flirt", "convert_xfm", "fslreorient2std"]
    return all(shutil.which(binary) for binary in required_binaries)

def atlas_registration(
    atlas_path: str,
    mni_template_path: str,
    image_path: str,      # MRA
    image_t1: str,        # T1
    output_dir: str = "."
) -> str:
    
    if not _fsl_available():
        raise RuntimeError("FSL is not installed or not in PATH.")

    # File naming setup
    t1_name = os.path.basename(image_t1).split('.')[0]
    
    # 0. ROBUST FOV (NECK CROP)
    logger.debug(f"  [DEBUG 0/6] ROBUSTFOV: Removing neck tissue...")
    t1_fov = os.path.join(output_dir, f"{t1_name}_cropped_fov.nii.gz")
    rfov = fsl.RobustFOV(in_file=image_t1, out_roi=t1_fov, brainsize=180)
    rfov.run()

    # 1. BRAIN EXTRACTION
    logger.debug(f"  [DEBUG 1/6] BET: Extracting brain...")
    t1_brain = os.path.join(output_dir, f"{t1_name}_BETted_brain.nii.gz")
    bet = fsl.BET(in_file=t1_fov, out_file=t1_brain, mask=True, frac=0.4)
    bet.run()

    # 2. REORIENT TO STANDARD
    logger.debug(f"  [DEBUG 2/6] REORIENT: Aligning T1 orientation to standard...")
    t1_reoriented = os.path.join(output_dir, f"{t1_name}_Reoriented.nii.gz")
    reorient = fsl.Reorient2Std(in_file=t1_brain, out_file=t1_reoriented)
    reorient.run()

    # 3. T1 -> MRA (6-DOF)
    # We save the image here so you can check if T1 and MRA overlap correctly
    logger.debug(f"  [DEBUG 3/6] FLIRT: Registering Subject T1 to Subject MRA...")
    t1_in_mra_img = os.path.join(output_dir, f"{t1_name}_T1_in_MRA.nii.gz")
    t1_to_mra_mat = os.path.join(output_dir, "t1_to_mra.mat")
    
    flirt_t1_mra = fsl.FLIRT()
    flirt_t1_mra.inputs.in_file = t1_reoriented
    flirt_t1_mra.inputs.reference = image_path
    flirt_t1_mra.inputs.out_file = t1_in_mra_img
    flirt_t1_mra.inputs.out_matrix_file = t1_to_mra_mat 
    flirt_t1_mra.inputs.dof = 6
    flirt_t1_mra.inputs.cost = 'mutualinfo'
    flirt_t1_mra.run()

    # 4. MNI -> T1 (12-DOF)
    # We save the image here to see if the MNI Template fits your patient's T1
    logger.debug(f"  [DEBUG 4/6] FLIRT: Registering MNI Template to Subject T1...")
    mni_in_t1_img = os.path.join(output_dir, f"{t1_name}_MNI_in_T1.nii.gz")
    mni_to_t1_mat = os.path.join(output_dir, "mni_to_t1.mat")
    
    flirt_mni_t1 = fsl.FLIRT()
    flirt_mni_t1.inputs.in_file = mni_template_path
    flirt_mni_t1.inputs.reference = t1_reoriented
    flirt_mni_t1.inputs.out_file = mni_in_t1_img
    flirt_mni_t1.inputs.out_matrix_file = mni_to_t1_mat
    flirt_mni_t1.inputs.dof = 12
    flirt_mni_t1.inputs.cost = 'corratio'
    flirt_mni_t1.run()

    # 5. CONCATENATE MATRICES
    logger.debug(f"  [DEBUG 5/6] CONVERT_XFM: Combining transforms (MNI -> T1 -> MRA)...")
    combined_mat = os.path.join(output_dir, "combined_mni_to_mra.mat")
    concat = fsl.ConvertXFM()
    concat.inputs.in_file = mni_to_t1_mat
    concat.inputs.in_file2 = t1_to_mra_mat
    concat.inputs.concat_xfm = True
    concat.inputs.out_file = combined_mat
    concat.run()

    # 6. FINAL WARP: ATLAS -> MRA
    logger.debug(f"  [DEBUG 6/6] APPLYXFM: Warping Atlas labels to MRA space...")
    reg_atlas_path = os.path.join(output_dir, f"{t1_name}_registered_atlas.nii.gz")
    apply_xfm = fsl.ApplyXFM()
    apply_xfm.inputs.in_file = atlas_path
    apply_xfm.inputs.reference = image_path
    apply_xfm.inputs.apply_xfm = True
    apply_xfm.inputs.in_matrix_file = combined_mat 
    apply_xfm.inputs.interp = 'nearestneighbour'
    apply_xfm.inputs.out_file = reg_atlas_path
    apply_xfm.run()

    return reg_atlas_path

def process_registration(image_path, image_t1_path, mask_path, atlas_path, output_dir, mni_template_path=None):
    # If no path was provided via CLI, use the default hardcoded guessing logic
    if mni_template_path is None:
        atlas_dir = os.path.dirname(atlas_path)
        # Try both common casings
        potential_path = os.path.join(atlas_dir, "MNI152_T1_1mm_brain.nii.gz")
        if not os.path.exists(potential_path):
            potential_path = os.path.join(atlas_dir, "MNI152_T1_1mm_Brain.nii.gz")
        mni_template_path = potential_path

    # Final check
    if not os.path.exists(mni_template_path):
        raise FileNotFoundError(f"MNI Template not found at: {mni_template_path}")

    # Pass the confirmed path to the registration function
    atlas_registration(atlas_path, mni_template_path, image_path, image_t1_path, output_dir)