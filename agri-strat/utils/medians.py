def get_medians_subpatch_path(patch_dir, subpatch_id, num_subpatches_per_patch, labels=False):
    subpatch_str = str(subpatch_id).rjust(len(str(num_subpatches_per_patch)), "0")
    return patch_dir / f'sub{subpatch_str}{"_labels" if labels else ""}.npy'
