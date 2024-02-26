def get_medians_subpatch_path(patch_dir, subpatch_id, num_subpatches, labels=False):
    return patch_dir / f'sub{str(subpatch_id).rjust(len(str(num_subpatches)), "0")}{"_labels" if labels else ""}.npy'
