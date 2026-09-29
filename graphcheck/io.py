import numpy as np


def load_prof_dict_npz(filename):
    """
    Load a prof_dict saved with save_prof_dict_npz()

    Notes:
        - 0-D arrays are converted back to Python scalars
        - match_windows is reconstructed as a list of 2D arrays
    """

    z = np.load(filename, allow_pickle=False)
    keys = z["__keys__"].astype(np.int64)

    prof_dict = {}

    for pid in keys:
        pid = int(pid)
        entry = {}

        prefix = f"{pid}/"

        for name in z.files:
            if not name.startswith(prefix):
                continue

            if name.startswith(f"{pid}/match_windows/"):
                continue

            field = name[len(prefix):]
            arr = z[name]

            if arr.ndim == 0:
                entry[field] = arr.item()
            else:
                entry[field] = arr

        # reconstruct list of 2D arrays
        nname = f"{pid}/match_windows/n"
        if nname in z:
            n = int(z[nname].item())
            entry["match_windows"] = [
                z[f"{pid}/match_windows/{i}"].copy() for i in range(n)
            ]

        prof_dict[pid] = entry

    return prof_dict
