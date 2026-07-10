import os
import glob
import numpy as np
import cv2


class ThermalDataLoader:
    """
    Loads thermal images from disk and applies preprocessing:
      1. Read raw image (8-bit or 16-bit)
      2. Crop left half if stereo side-by-side
      3. Undistort (plumb-bob remap) when K, D are given
      4. Gaussian blur to denoise thermal sensor speckle
      5. Normalize 16-bit to 8-bit (percentile / fixed / 14bit)
      6. CLAHE for local contrast enhancement (on 8-bit)

    --- A/B VARIANT (dataloader2) ---------------------------------------
    This module is an exact copy of dataloader.py with ONE change: the order
    of the normalization and CLAHE stages is swapped, so CLAHE is applied
    *after* the 8-bit normalization (i.e. directly on the 8-bit image) rather
    than on the native bit-depth. This is the conventional ordering in the
    literature and makes the CLAHE clip_limit operate on a dense 256-bin
    histogram. Used to test preprocessing sensitivity (RQ3) on a single
    dataset; swap the import in the test script (dataloader -> dataloader2)
    to switch pipelines. Everything else is identical.
    ---------------------------------------------------------------------

    Supports SThereo and FIReStereo datasets.
    """

    def __init__(self, image_dir,
                 bit_depth=8,
                 is_stereo_side_by_side=False,
                 use_clahe=True,
                 clahe_clip=3.0,
                 clahe_grid=(8, 8),
                 normalize='percentile',
                 norm_range=None,
                 percentile=(2.0, 98.0),
                 undistort=False,
                 K=None,
                 D=None,
                 distortion_model='plumb_bob',
                 gaussian_sigma=0.0,
                 gaussian_ksize=(5, 5)):
        """
        Args:
            image_dir:              path to image folder
            bit_depth:              8 for SThereo (visible), 16 for thermal raw
            is_stereo_side_by_side: True for SThereo visible stereo (1280x560 → crop left)
            clahe_clip:             CLAHE clip limit (higher = more contrast, more noise)
            clahe_grid:             CLAHE tile grid size
            normalize:              'percentile' (default) → per-frame percentile clip,
                                    'fixed'                → use norm_range,
                                    '14bit'                → divide by 16383 (legacy).
            norm_range:             (low, high) tuple for normalize='fixed'
            percentile:             (low, high) percentiles for normalize='percentile'
            undistort:              when True, applies plumb-bob undistortion using K,D.
                                    The output keeps the same K, so K^-1 maps undistorted
                                    pixels to normalized bearings directly.
            K:                      3x3 camera matrix  (required if undistort=True)
            D:                      distortion coefficients (k1,k2,p1,p2,k3 — plumb_bob,
                                    OR k1,k2,k3,k4 — equidistant/fisheye)
            distortion_model:       'plumb_bob' (default) or 'equidistant' (fisheye).
                                    Selects between cv2.initUndistortRectifyMap and
                                    cv2.fisheye.initUndistortRectifyMap.
        """
        self.image_dir              = image_dir
        self.bit_depth              = bit_depth
        self.is_stereo_side_by_side = is_stereo_side_by_side
        self.use_clahe              = use_clahe
        self.normalize              = normalize
        self.norm_range             = norm_range
        self.percentile             = percentile
        self.undistort              = undistort
        self.K                      = K
        self.D                      = D
        self.distortion_model       = distortion_model
        self.gaussian_sigma         = float(gaussian_sigma)
        if distortion_model not in ('plumb_bob', 'equidistant'):
            raise ValueError(f"Unknown distortion_model: {distortion_model}")
        self.gaussian_ksize         = gaussian_ksize
        self._undistort_maps        = None   # built lazily on first frame

        if normalize not in ('percentile', 'fixed', '14bit'):
            raise ValueError(f"Unknown normalize mode: {normalize}")
        if normalize == 'fixed' and norm_range is None:
            raise ValueError("normalize='fixed' requires norm_range=(low, high)")
        if undistort and (K is None or D is None):
            raise ValueError("undistort=True requires K and D")

        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)

        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No PNG images found in: {image_dir}")

        self.timestamps = self._parse_timestamps()

    def _parse_timestamps(self):
        """
        Parse timestamps from filenames.

        SThereo: filename = nanosecond timestamp (e.g. 1630073626275932013.png)
        FIReStereo: filename = frame index (e.g. 01998.png), timestamps from separate file
        """
        names = [os.path.splitext(os.path.basename(p))[0] for p in self.image_paths]

        # If filenames are large numbers (>1e12), treat as nanosecond timestamps
        if int(names[0]) > 1e12:
            return np.array([int(n) / 1e9 for n in names])  # nanoseconds → seconds
        else:
            # Frame indices — caller must set timestamps externally via load_timestamps_file()
            return np.arange(len(names), dtype=np.float64)

    def load_timestamps_file(self, timestamps_path, unit='ms'):
        """
        Load timestamps from an external file (one timestamp per line).
        Used for FIReStereo where timestamps.txt maps frame index → time.

        Args:
            timestamps_path: path to timestamps file
            unit:            'ms' (milliseconds) or 's' (seconds)
        """
        all_ts = np.loadtxt(timestamps_path)
        scale  = 1e-3 if unit == 'ms' else 1.0

        names = [os.path.splitext(os.path.basename(p))[0] for p in self.image_paths]
        self.timestamps = np.array([all_ts[int(n)] * scale for n in names])

    def _load_single(self, path):
        """Load, preprocess, and return a single image as uint8 grayscale.
        Pipeline: read raw → (crop) → undistort → (gaussian blur) → normalize to 8-bit → CLAHE.
        Blur denoises the raw thermal image first; the 16-bit frame is then
        normalized to 8-bit, and CLAHE is applied on the 8-bit image so the
        clip_limit acts on a dense 256-bin histogram (conventional ordering).
        This is the A/B counterpart to dataloader.py, which applies CLAHE on
        the native bit-depth *before* normalization."""
        if self.bit_depth == 16:
            raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        else:
            raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if raw is None:
            raise FileNotFoundError(f"Could not read: {path}")

        # Crop left half for side-by-side stereo (e.g. SThereo visible-light stereo)
        if self.is_stereo_side_by_side:
            h, w = raw.shape[:2]
            raw = raw[:, :w // 2]

        # Undistort using a precomputed remap LUT — keeps K identical, so callers
        # can apply K^-1 directly to feature pixels for normalized bearings.
        # plumb_bob: cv2.initUndistortRectifyMap (k1,k2,p1,p2,k3 distortion).
        # equidistant (fisheye): cv2.fisheye.initUndistortRectifyMap (k1..k4).
        if self.undistort:
            if self._undistort_maps is None:
                h, w = raw.shape[:2]
                if self.distortion_model == 'equidistant':
                    self._undistort_maps = cv2.fisheye.initUndistortRectifyMap(
                        self.K, self.D.reshape(4, 1), np.eye(3), self.K,
                        (w, h), cv2.CV_16SC2)
                else:   # plumb_bob
                    self._undistort_maps = cv2.initUndistortRectifyMap(
                        self.K, self.D, None, self.K, (w, h), cv2.CV_16SC2)
            raw = cv2.remap(raw, *self._undistort_maps, cv2.INTER_LINEAR)

        # Gaussian blur on the raw bit-depth — denoises thermal sensor speckle
        # before any contrast enhancement has a chance to amplify it.
        if self.gaussian_sigma > 0.0:
            raw = cv2.GaussianBlur(raw, self.gaussian_ksize, self.gaussian_sigma)

        # 8-bit normalization FIRST (A/B change). For 16-bit input this maps the
        # native radiometric range into 0..255 so that the subsequent CLAHE
        # operates on a dense 256-bin histogram and its clip_limit is meaningful.
        if self.bit_depth == 16:
            if self.normalize == '14bit':
                lo, hi = 0.0, 16383.0
            elif self.normalize == 'fixed':
                lo, hi = self.norm_range
            else:  # 'percentile'
                lo, hi = np.percentile(raw, self.percentile) # 2% and 98% by default
                if hi - lo < 1.0:           # near-uniform frame, fall back to full range
                    lo, hi = float(raw.min()), float(raw.max() + 1)

            raw = ((raw.astype(np.float32) - lo) / (hi - lo) * 255
                   ).clip(0, 255).astype(np.uint8)

        # CLAHE on the 8-bit image (A/B change: applied AFTER normalization).
        if self.use_clahe:
            raw = self.clahe.apply(raw)

        return raw

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Returns (image, timestamp) for the given index."""
        return self._load_single(self.image_paths[idx]), self.timestamps[idx]

    def __iter__(self):
        """Iterate over all (image, timestamp) pairs."""
        for path, ts in zip(self.image_paths, self.timestamps):
            yield self._load_single(path), ts

class IMULoader:
    """
    Loads IMU data from a text/CSV file and projects it into the canonical
    layout [t, gx, gy, gz, ax, ay, az] (gyro rad/s, accel m/s^2, t seconds).

    The source file may have any number of extra columns (quaternion, Euler,
    magnetometer, etc.); only the time / gyro / accel columns are kept.

    Args:
        imu_file:    path to IMU file
        delimiter:   None (whitespace), ',' for CSV, etc.
        skip_header: header lines to skip
        time_col:    column index of timestamp
        gyro_cols:   3-tuple of column indices for (gx, gy, gz)
        accel_cols:  3-tuple of column indices for (ax, ay, az)
        t_offset:    added to every IMU timestamp to align IMU clock with
                     the camera clock (t_imu_in_cam = t_imu + t_offset).
    """

    def __init__(self, imu_file, delimiter=None, skip_header=0,
                 time_col=0, gyro_cols=(1, 2, 3), accel_cols=(4, 5, 6),
                 t_offset=0.0):
        if not os.path.isfile(imu_file):
            raise FileNotFoundError(f"IMU file not found: {imu_file}")

        raw = np.loadtxt(imu_file, delimiter=delimiter, skiprows=skip_header)

        # Project to canonical [t, gx, gy, gz, ax, ay, az]
        cols = [time_col, *gyro_cols, *accel_cols]
        data = raw[:, cols].astype(np.float64)

        # Ensure chronological order
        data = data[np.argsort(data[:, 0])]

        # Auto-convert nanosecond timestamps → seconds
        if data[0, 0] > 1e12:
            data[:, 0] /= 1e9

        data[:, 0] += t_offset

        self.data       = data
        self.timestamps = data[:, 0]
        self.gyro       = data[:, 1:4]
        self.accel      = data[:, 4:7]

    @classmethod
    def sthereo(cls, imu_file, t_offset=0.0):
        """
        SThereo xsens_imu.csv layout:
          t, qx, qy, qz, qw, ex, ey, ez, gx, gy, gz, ax, ay, az, mx, my, mz
        """
        return cls(imu_file, delimiter=',', skip_header=0,
                   time_col=0, gyro_cols=(8, 9, 10), accel_cols=(11, 12, 13),
                   t_offset=t_offset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        for row in self.data:
            yield row

    def slice(self, t_start, t_end):
        """Return IMU rows in the half-open interval (t_start, t_end]."""
        i0 = np.searchsorted(self.timestamps, t_start, side='right')
        i1 = np.searchsorted(self.timestamps, t_end,   side='right')
        return self.data[i0:i1]


class VIOSequencer:
    """
    Synchronizes a ThermalDataLoader with an IMULoader.

    Two iteration modes:
      • Batch (default __iter__): yields (img, t_cam, imu_batch) where
        imu_batch contains all IMU rows in (t_{k-1}, t_k]. The first batch
        also includes IMU samples before the first camera frame, which is
        useful for initial gravity / attitude estimation.

      • Event (`.events()`): yields ('imu', t, row) and ('cam', t, img) in
        strict chronological order — closer to a live sensor stream.
    """

    def __init__(self, thermal_loader, imu_loader):
        self.thermal_loader = thermal_loader
        self.imu_loader     = imu_loader

    def __len__(self):
        return len(self.thermal_loader)

    def __iter__(self):
        prev_t = -np.inf
        for img, t_cam in self.thermal_loader:
            imu_batch = self.imu_loader.slice(prev_t, t_cam)
            yield img, t_cam, imu_batch
            prev_t = t_cam

    def events(self):
        ts_imu = self.imu_loader.timestamps
        ts_cam = self.thermal_loader.timestamps
        n_imu, n_cam = len(ts_imu), len(ts_cam)
        i = j = 0

        while i < n_imu or j < n_cam:
            take_imu = (j == n_cam) or (i < n_imu and ts_imu[i] <= ts_cam[j])
            if take_imu:
                yield 'imu', ts_imu[i], self.imu_loader.data[i]
                i += 1
            else:
                img, _ = self.thermal_loader[j]
                yield 'cam', ts_cam[j], img
                j += 1
