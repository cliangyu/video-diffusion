import h5py

filename = "/mnt/lustre/share/boli/ucf101_64px_video_gen/train.h5"

with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys)
    # these can be group or dataset names
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[0]
    pass
