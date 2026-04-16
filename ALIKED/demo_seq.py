import copy
import os
import time
import cv2
import glob
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from nets.aliked import ALIKED

class ImageLoader(object):
    def __init__(self, filepath: str):
        self.N = 3000
        if filepath.startswith('camera'):
            camera = int(filepath[6:])
            self.cap = cv2.VideoCapture(camera)
            if not self.cap.isOpened():
                raise IOError(f"Can't open camera {camera}!")
            logging.info(f'Opened camera {camera}')
            self.mode = 'camera'
        elif os.path.exists(filepath):
            if os.path.isfile(filepath):
                self.cap = cv2.VideoCapture(filepath)
                if not self.cap.isOpened():
                    raise IOError(f"Can't open video {filepath}!")
                rate = self.cap.get(cv2.CAP_PROP_FPS)
                self.N = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
                duration = self.N / rate
                logging.info(f'Opened video {filepath}')
                logging.info(f'Frames: {self.N}, FPS: {rate}, Duration: {duration}s')
                self.mode = 'video'
            else:
                self.images = glob.glob(os.path.join(filepath, '*.png')) + \
                              glob.glob(os.path.join(filepath, '*.jpg')) + \
                              glob.glob(os.path.join(filepath, '*.ppm'))
                self.images.sort()
                self.N = len(self.images)
                logging.info(f'Loading {self.N} images')
                self.mode = 'images'
        else:
            raise IOError('Error filepath (camerax/path of images/path of videos): ', filepath)

    def __getitem__(self, item):
        if self.mode == 'camera' or self.mode == 'video':
            if item > self.N:
                return None
            ret, img = self.cap.read()
            if not ret:
                raise "Can't read image from camera"
            if self.mode == 'video':
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, item)
        elif self.mode == 'images':
            filename = self.images[item]
            img = cv2.imread(filename)
            if img is None:
                raise Exception('Error reading image %s' % filename)        
        return img

    def __len__(self):
        return self.N


def resize_by_max_edge(img, max_edge):
    if max_edge <= 0:
        return img, 1.0

    height, width = img.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    if scale == 1.0:
        return img, scale

    new_size = (int(round(width * scale)), int(round(height * scale)))
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


class SimpleTracker(object):
    def __init__(self):
        self.pts_prev = None
        self.desc_prev = None

    def update(self, img, pts, desc):
        N_matches = 0
        if self.pts_prev is None:
            self.pts_prev = pts
            self.desc_prev = desc

            out = copy.deepcopy(img)
            for pt1 in pts:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 1, (0, 0, 255), -1, lineType=16)
        else:
            matches = self.mnn_mather(self.desc_prev, desc)
            mpts1, mpts2 = self.pts_prev[matches[:, 0]], pts[matches[:, 1]]
            N_matches = len(matches)

            out = copy.deepcopy(img)
            for pt1, pt2 in zip(mpts1, mpts2):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, (0, 255, 0), lineType=16)
                cv2.circle(out, p2, 1, (0, 0, 255), -1, lineType=16)

            self.pts_prev = pts
            self.desc_prev = desc

        return out, N_matches

    def mnn_mather(self, desc1, desc2):
        sim = desc1 @ desc2.transpose()
        sim[sim < 0.9] = 0
        nn12 = np.argmax(sim, axis=1)
        nn21 = np.argmax(sim, axis=0)
        ids1 = np.arange(0, sim.shape[0])
        mask = (ids1 == nn21[nn12])
        matches = np.stack([ids1[mask], nn12[mask]])
        return matches.transpose()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ALIKED sequence Demo.')
    parser.add_argument('input', type=str, default='',
                        help='Image directory or movie file or "camera0" (for webcam0).')
    parser.add_argument('--model', choices=['aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'], default="aliked-n32",
                        help="The model configuration")
    parser.add_argument('--device', type=str, default='cuda', help="Running device (default: cuda, falls back to cpu if unavailable).")
    parser.add_argument('--top_k', type=int, default=-1,
                        help='Detect top K keypoints. -1 for threshold based mode, >0 for top K mode. (default: -1)')
    parser.add_argument('--scores_th', type=float, default=0.2,
                        help='Detector score threshold (default: 0.2).')
    parser.add_argument('--n_limit', type=int, default=5000,
                        help='Maximum number of keypoints to be detected (default: 5000).')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save result images (default: output).')
    parser.add_argument('--max_edge', type=int, default=1600,
                        help='Resize input so the longest edge is at most this value before inference. Use <=0 to disable. (default: 1600)')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile to optimize the model.')
    parser.add_argument('--n_frames', type=int, default=0,
                        help='Max number of frames to process. 0 for all. (default: 0)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    image_loader = ImageLoader(args.input)
    model = ALIKED(model_name=args.model,
                  device=args.device,
                  top_k=args.top_k,
                  scores_th=args.scores_th,
                  n_limit=args.n_limit)
    if args.compile:
        model = torch.compile(model, mode='reduce-overhead')
        logging.info('Using torch.compile (reduce-overhead mode)')
    tracker = SimpleTracker()

    os.makedirs(args.output, exist_ok=True)
    logging.info(f'Saving results to: {os.path.abspath(args.output)}')

    runtime_extract = []
    runtime_match = []
    resize_logged = False
    progress_bar = tqdm(image_loader)
    for frame_idx, img in enumerate(progress_bar):
        if img is None:
            break
        if args.n_frames > 0 and frame_idx >= args.n_frames:
            break
        img, resize_scale = resize_by_max_edge(img, args.max_edge)
        if not resize_logged:
            logging.info(f'Inference image size: {img.shape[1]}x{img.shape[0]} (scale={resize_scale:.3f})')
            resize_logged = True
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = model.run(img_rgb)
        kpts = pred['keypoints']
        desc = pred['descriptors']
        runtime_extract.append(pred['time'])

        t_match0 = time.time()
        out, N_matches = tracker.update(img, kpts, desc)
        t_match1 = time.time()
        runtime_match.append(t_match1 - t_match0)

        ave_extract_ms = np.mean(runtime_extract) * 1000
        ave_match_ms = np.mean(runtime_match) * 1000
        status = (f"extract:{ave_extract_ms:.1f}ms  match:{ave_match_ms:.1f}ms  "
                  f"matches/kpts:{N_matches}/{len(kpts)}")
        progress_bar.set_description(status)

        score_map = (pred['score_map']*255).astype(np.uint8)
        score_map_colorjet = cv2.applyColorMap(score_map, cv2.COLORMAP_JET)
        vis_img = np.hstack((out, score_map_colorjet))
        save_path = os.path.join(args.output, f'frame_{frame_idx:06d}.jpg')
        cv2.imwrite(save_path, vis_img)

    logging.info('Finished!')
    if len(runtime_extract) > 0:
        logging.info(
            f"[Timing summary over {len(runtime_extract)} frames]\n"
            f"  Feature extraction : mean={np.mean(runtime_extract)*1000:.1f}ms  "
            f"median={np.median(runtime_extract)*1000:.1f}ms  "
            f"min={np.min(runtime_extract)*1000:.1f}ms  "
            f"max={np.max(runtime_extract)*1000:.1f}ms\n"
            f"  Feature matching   : mean={np.mean(runtime_match)*1000:.1f}ms  "
            f"median={np.median(runtime_match)*1000:.1f}ms  "
            f"min={np.min(runtime_match)*1000:.1f}ms  "
            f"max={np.max(runtime_match)*1000:.1f}ms"
        )
    logging.info(f'Results saved to: {os.path.abspath(args.output)}')