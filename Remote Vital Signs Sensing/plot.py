import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import h5py
import torch
from PIL import Image

'python plot.py -d "../data/UBFCChunks72x72_BackgroundAug_Green/ ../data/UBFCChunks72x72_BackgroundAug_Black" -p "../results/Original_vs_Black_bg/original/waveform/ ../results/Original_vs_Black_bg/black/waveform/"'
" python plot.py -d '/home/patrick/CVProjects/MTTS-CAN/EfficientPhys/results/att_masks/Black /home/patrick/CVProjects/MTTS-CAN/EfficientPhys/results/att_masks/Original' -o /home/patrick/CVProjects/MTTS-CAN/EfficientPhys/results/att_masks/new_combined"
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--dir_names', default="*", type=str, help="Names of the sub-directories to process, default setting: all folders. Format: 'folder1 folder2' ")
parser.add_argument('-d', '--data_dirs', required=True, type=str, help="Data directories, for multiple inputs, add quotes like:"
                                                        " './a ./b'")
parser.add_argument('-c', '--chunk', action='store_true', help="True if there exist subfolder for each subject, False for whole-video results")
parser.add_argument('-o', '--output_dir', required=True, type=str, help="Output directory")
# parser.add_argument('-vid', '--video', action='store_true', help="Output video instead of image")
args = parser.parse_args()


def extract(data_path):
    with h5py.File(data_path, 'r') as f:
        data = torch.tensor(np.array(f['dXsub']))
        data = data.permute(3, 2, 1, 0).detach().cpu().numpy()
        return data[..., 0:3], data[..., 3:6], np.array(f['dysub']), np.array(f['drsub'])


def extract_normalized():
    data_dir = args.data_dirs.split(" ")[0]
    dir_names = os.listdir(data_dir)

    for dir_name in dir_names:
        dir_path = os.path.join(data_dir, dir_name)
        file_names = os.listdir(os.path.join(data_dir, dir_name))
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)
            snapshot_norm, _, _, _ = extract(file_path)

            # Normalize value to 0~255
            # snapshot_norm = snapshot_norm - np.amin(snapshot_norm)
            # snapshot_norm = np.uint8(snapshot_norm * 250 / np.amax(snapshot_norm))
            # print(np.amax(snapshot_norm), np.amin(snapshot_norm))
            plt.imshow(snapshot_norm[0])
            plt.savefig(os.path.join(args.output_dir, file_name[:-4] + '.png'),
                        dpi=200)
            plt.close()

def main():

    data_dirs = args.data_dirs.split(" ")

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    n_plot_rows = len(data_dirs)

    if args.chunk:
        if args.dir_names == "*":
            dir_names = set(os.listdir(data_dirs[0]))
        else:
            dir_names = set(args.dir_names.split(' '))
        for i in range(1, n_plot_rows):
            dir_names = set(dir_names).intersection(os.listdir(data_dirs[i]))

        for dir_name in dir_names:
            file_names = os.listdir(os.path.join(data_dirs[0], dir_name) + "/snapshots")
            for file_name in file_names:
                final_img = None
                for i in range(n_plot_rows):
                    dir_path = os.path.join(data_dirs[i], dir_name)
                    snapshot_path = os.path.join(dir_path, "snapshots") + "/" + file_name
                    snapshot_norm_path = os.path.join(dir_path, "normalized") + "/" + file_name
                    attn_mask_path = os.path.join(dir_path, "attn_masks") + "/" + file_name
                    waveform_path = os.path.join(dir_path, "waveforms") + "/" + file_name

                    # Combine Snapshot and Attention Mask
                    snapshot = Image.open(snapshot_path)
                    snapshot_norm = Image.open(snapshot_norm_path)
                    attn_mask = Image.open(attn_mask_path)
                    waveform = Image.open(waveform_path)
                    if final_img is None:
                        final_img = Image.new('RGB', (snapshot.size[0] * 3 + waveform.size[0], max(snapshot.size[1], waveform.size[1]) * n_plot_rows), (250, 250, 250))
                    combined_img = Image.new('RGB', (snapshot.size[0] * 3 + waveform.size[0], max(snapshot.size[1], waveform.size[1])), (250, 250, 250))
                    combined_img.paste(snapshot, (0, 0))
                    combined_img.paste(snapshot_norm, (snapshot.size[0], 0))
                    combined_img.paste(attn_mask, (snapshot.size[0] * 2, 0))
                    combined_img.paste(waveform, (snapshot.size[0] * 3, 0))
                    final_img.paste(combined_img, (0, i * max(snapshot.size[1], waveform.size[1])))
                final_img.save(os.path.join(args.output_dir, file_name))
    else:
        dir_name = data_dirs[0]
        file_names = os.listdir(os.path.join(dir_name, "snapshots"))
        for file_name in file_names:
            final_img = None
            for i in range(n_plot_rows):
                dir_path = data_dirs[i]
                snapshot_path = os.path.join(dir_path, "snapshots") + "/" + file_name
                snapshot_norm_path = os.path.join(dir_path, "normalized") + "/" + file_name
                attn_mask_path = os.path.join(dir_path, "attn_masks") + "/" + file_name
                waveform_path = os.path.join(dir_path, "waveforms") + "/" + file_name

                # Combine Snapshot and Attention Mask
                snapshot = Image.open(snapshot_path)
                snapshot_norm = Image.open(snapshot_norm_path)
                attn_mask = Image.open(attn_mask_path)
                waveform = Image.open(waveform_path)
                if final_img is None:
                    final_img = Image.new('RGB', (snapshot.size[0] * 3 + waveform.size[0], max(snapshot.size[1], waveform.size[1]) * n_plot_rows), (250, 250, 250))
                combined_img = Image.new('RGB', (snapshot.size[0] * 3 + waveform.size[0], max(snapshot.size[1], waveform.size[1])), (250, 250, 250))
                combined_img.paste(snapshot, (0, 0))
                combined_img.paste(snapshot_norm, (snapshot.size[0], 0))
                combined_img.paste(attn_mask, (snapshot.size[0] * 2, 0))
                combined_img.paste(waveform, (snapshot.size[0] * 3, 0))
                final_img.paste(combined_img, (0, i * max(snapshot.size[1], waveform.size[1])))
            final_img.save(os.path.join(args.output_dir, file_name))


if __name__ == "__main__":
    main()
