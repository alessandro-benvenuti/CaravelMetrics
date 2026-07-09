import os
import vedo
import nibabel as nib

def test_volume_render(volume_path):
    if not os.path.exists(volume_path):
        print(f"File not found: {volume_path}")
        return

    print(f"Loading {volume_path}...")
    nii = nib.load(volume_path)
    
    # 1. Create the Volume
    vol = vedo.Volume(nii.get_fdata())
    vol.apply_transform(nii.affine.tolist())
    
    # Render settings for the cloudy volume
    vol.cmap("bone").alpha([0, 0.2, 0.5, 0.8])
    
    # 2. Create solid orthogonal slices (This DOES NOT require GPU ray-casting)
    # It cuts through the middle of the X, Y, and Z axes
    slices = vol.slice_orthogonal()
    
    # Show both. If GPU volume rendering fails, you will still see the slices!
    plt = vedo.Plotter(title="Volume Test", bg="blackboard")
    plt.show([vol, slices], axes=1)

if __name__ == "__main__":
    # Change this to your exact MNI template path
    mni_path = "Atlas/MNI152_T1_1mm_Brain.nii.gz" 
    
    # Fallback to look around if the path is wrong
    if not os.path.exists(mni_path):
        mni_path = input("Enter the path to your MNI template .nii.gz: ")

    test_volume_render(mni_path)