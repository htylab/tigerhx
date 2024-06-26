import nibabel as nib
import numpy as np
import os

def load_2d_nifti(file_path):
    nii = nib.load(file_path)
    data = nii.get_fdata()
    affine = nii.affine
    return data, affine

def save_3d_nifti(data, output_path, affine):
    new_nii = nib.Nifti1Image(data, affine)
    nib.save(new_nii, output_path)

def main(input_folder, output_file, slice_gap):
    # 假設所有2D NIfTI檔案在同一個資料夾中，且按順序命名
    nii_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.nii') or f.endswith('.nii.gz')])

    # 讀取第一個檔案以獲取形狀資訊和仿射矩陣
    first_data, first_affine = load_2d_nifti(os.path.join(input_folder, nii_files[0]))

    # 初始化3D數組
    num_slices = len(nii_files)
    shape_3d = (first_data.shape[0], first_data.shape[1], num_slices)
    data_3d = np.zeros(shape_3d)

    # 讀取每個2D檔案並插入3D數組
    for i, nii_file in enumerate(nii_files):
        slice_data, _ = load_2d_nifti(os.path.join(input_folder, nii_file))
        data_3d[:, :, i] = slice_data

    # 調整仿射矩陣
    new_affine = first_affine.copy()
    new_affine[2, 2] = slice_gap  # 設定切片間隙
    new_affine[:2, 2] = 0  # 確保在Z軸上的偏移是正確的
    new_affine[2, 3] = first_affine[2, 3]  # 確保在Z軸上的原點位置是正確的

    # 儲存3D NIfTI檔案
    save_3d_nifti(data_3d, output_file, new_affine)

if __name__ == "__main__":
    input_folder = "path/to/2d/nii/folder"  # 替換成2D NIfTI檔案所在的資料夾路徑
    output_file = "path/to/output/3d_nifti.nii"  # 替換成要儲存的3D NIfTI檔案路徑
    slice_gap = 1.0  # 設定切片間隙，根據實際情況進行調整
    main(input_folder, output_file, slice_gap)
