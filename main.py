import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
import sys
from pathlib import Path
from scipy.ndimage import zoom, map_coordinates
from scipy.signal import fftconvolve
from typing import Literal, TypeAlias

# –––––––––––––––––– Defining Class ––––––––––––––––––


class DoseMap:

    def __init__(
        self, SSD, beam_width0, patient_width, kernal_pixel_width, mu, resolution
    ):
        self.DPB_kernals = {
            1.25: np.rot90(self._read_txt_arr("kernel_2D_1250keV"), -1),
            10: np.rot90(self._read_txt_arr("kernel_2D_10MeV"), -1),
            20: np.rot90(self._read_txt_arr("kernel_2D_20MeV"), -1),
        }  # MeV: 161 x 161 mm^2
        self.mu = mu
        self.SSD = SSD
        self.beam_width0 = beam_width0
        self.patient_width = patient_width
        self.kernal_pixel_width = kernal_pixel_width
        self.patient_mask = self._read_txt_arr(
            "patient_surface"
        )  # 512 x 512, 15 x 15 cm^2
        self.resolution = resolution
        self.N = self.patient_mask.shape[
            1
        ]  # N = Nx = Ny (since it's a square) for patient
        self.M = self.DPB_kernals[1.25].shape[1]

        # Initialize dose map with zeros. We create a dose map for each energy
        self.dose_maps = {
            1.25: np.zeros((self.N, self.N)),
            10: np.zeros((self.N, self.N)),
            20: np.zeros((self.N, self.N)),
        }

        # Initialize Dose DPB product
        self.DPB_Dose = {}
        # Initialize DPB_polor maps

        # Find SSD point on central dose map coordinate system:
        self.x0_idx = self.y0_idx = int(
            (self.N - 1) / 2
        )  # x (or y) idx of central axis

        # Use 1D array of surface y idx's
        self.surface_arr = self._find_surface_arr()

        self.ySSD_cm = (self.y0_idx - (self.surface_arr[self.x0_idx] - 1 / 2)) * (
            self.patient_width / self.N
        )

        # central coordinate system for plotting (origin at SSD of central axis)
        self.x0_idx = int(self.N / 2 - 1)
        self.y0_idx = int(self.surface_arr[self.x0_idx])

        # Used across all plots to define x and y axis scales
        edge_cm = self.patient_width / 2  # 7.5 cm
        x_min = -edge_cm
        x_max = edge_cm
        y_max = edge_cm - self.ySSD_cm
        y_min = -edge_cm - self.ySSD_cm
        self.extent = [x_min, x_max, y_min, y_max]

    def create_fluence(self, energy: float) -> None:
        """
        Uniform beam profile, and (assumed) monoenergetic energy. Value decreases with distance due to ISL. Width increases due to divergence as described by similar triangles, for 6cm width at SSD = 100.

        Note since each Fluence = N * E / Area and each plot is scaled to be fluence 1 at SSD 100cm, the energy will not affect the fluence plot. Thus, N, E, and Area are ignored, and only ISL (affecting fluence value) and beam divergence (in terms of beam width) factors are accounted for.

        Note since the ISL factor is defined to be 1 at SSD 100, the relative beam fluence is by definition 1 at SSD 100 (as coherent with suggested method).

        The beam front is taken as a plane for simplicity (i.e. uniform profile).

        """
        fluence_map = np.zeros(self.patient_mask.shape)
        for y_idx in range(self.N):

            # Apply ISL:
            y_cm = self._y_cm(y_idx)
            dSSD_cm = self.ySSD_cm - y_cm
            SPD = dSSD_cm + self.SSD

            beam = (self.SSD / (SPD)) ** 2  # Apply ISL to values

            fluence_map[y_idx, :] = beam

            # Apply width divergence of beam:
            # Calculate range of x values for which there is a beam width
            beam_width_cm = (
                (SPD) * (self.beam_width0) / self.SSD
            )  # similar triangles, 100/6 = SPD/width, or width = SPD *  (width0/SSD)
            beam_width_idx = beam_width_cm * (self.N) / self.patient_width
            min_x_idx = self.x0_idx - round(beam_width_idx / 2)
            max_x_idx = self.x0_idx + round(beam_width_idx / 2)

            # Set x values outside of beam width to 0:
            for x_idx in range(self.N):
                if x_idx < min_x_idx or x_idx > max_x_idx:
                    fluence_map[y_idx, x_idx] = 0

        # Store flunence without mask for visualization
        self.fluence_wo_mask = fluence_map

        # Apply Patient Mask:
        self.dose_maps[energy] = fluence_map * self.patient_mask

    def apply_attenuation(
        self, energy: float, option: Literal["ray trace", "uniform"] = "ray trace"
    ) -> None:
        attenuation_map = self.patient_mask.copy()
        for y_idx in range(attenuation_map.shape[0]):
            for x_idx in range(attenuation_map.shape[1]):
                if attenuation_map[y_idx, x_idx]:  # if point P is within patient tissue
                    t = self._t((x_idx, y_idx), option)
                    attenuation_map[y_idx, x_idx] = math.exp(-self.mu[energy] * t)
                else:
                    attenuation_map[y_idx, x_idx] = 0
        self.dose_maps[energy] *= attenuation_map

    def apply_DPB_kernel(self, energy, within_patient=True) -> None:

        scatter_map = self._create_polar_DPB(resolution, energy)

        min_y_idx = int(
            self.surface_arr.min()
        )  # Where y idx less than this value are 0 for all x. This will save some time looping

        # Find min and max x idx to save computation:
        min_x_idx, max_x_idx = self._x_idx_bounds(energy)

        dose_DPB_map = np.zeros((self.N, self.N))
        cnt = 0
        tolerance = 50

        dtheta = resolution["dtheta"]
        dr = resolution["dr"]
        rmax = resolution["rmax"]

        last_percent = 0
        for y_idx in range(self.N):
            for x_idx in range(
                self.N
            ):  # Calculate dose close to non zero values, + a tolerance of 20 pixels
                # for y_idx in range(min_y_idx - tolerance, self.N):
                #     for x_idx in range(
                #         min_x_idx - tolerance, max_x_idx + tolerance
                #     ):  # Calculate dose close to non zero values, + a tolerance of 20 pixels
                ##### Percent Bar #####
                if within_patient and not self.patient_mask[y_idx, x_idx]:
                    continue
                cnt += 1
                percent = round(cnt / (149960) * 100, 2)
                if percent != last_percent:
                    sys.stdout.write(f"\rEnergy: {energy} MeV | Progress: {percent}%")
                    last_percent = percent
                sys.stdout.flush()
                #######################
                for theta in range(0, 360, dtheta):
                    for r in np.arange(0, rmax, dr):
                        x_prime_idx, y_prime_idx = self._x_y_prime_idx(
                            r, theta, x_idx, y_idx
                        )
                        # Main equation. Note it omitts mass-attenuation, therefore is relative dose. D(x,y) ~ integral { Fluence(x', y') * DPB(x', y') * exp{- mu * t} } dA. Note self.dose_maps includes both fluence and attenuation at this point in calculatino
                        if 0 <= x_prime_idx <= (self.N - 1) and 0 <= y_prime_idx <= (
                            self.N - 1
                        ):  # Check if in bounds
                            dose_DPB_map[y_idx, x_idx] += (
                                scatter_map[(r, theta)]
                                * self.dose_maps[energy][y_prime_idx, x_prime_idx]
                                * r
                                * dr
                                * dtheta
                            )
        self.dose_maps[energy] = dose_DPB_map

    def apply_DPB_conv(self, energy: float) -> None:
        image = self.dose_maps[energy]
        kernel = self.DPB_kernals[energy]
        dx_img = 15 / 512
        dx_k = 0.1

        scale = dx_k / dx_img  # kernel pixel size / image pixel size
        kernel_resampled = zoom(kernel, zoom=scale, order=1)  # linear interp

        self.dose_maps[energy] = fftconvolve(image, kernel_resampled, mode="same")

    def plot_DPB_kernal(self, energy: float) -> None:
        kernal = self.DPB_kernals[energy]
        # Rotate kernal so beam enters from top to bottom:
        # kernal = np.rot90(kernal_0, -1)
        edge_cm = (self.M + 1) / 2 * self.kernal_pixel_width
        y_max = x_max = edge_cm
        y_min = x_min = -edge_cm

        im = plt.imshow(
            kernal,
            cmap="jet",
            extent=[x_min, x_max, y_min, y_max],
            origin="upper",
            norm=LogNorm(),
        )

        cbar = plt.colorbar(im)
        cbar.set_label("Fractional Dose Delivered")
        plt.title(f"{energy} MeV DPB Kernel")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Distance (cm)")
        plt.show()

    def plot_dose_map(
        self,
        energy: float,
        title: str = "2D Dose Map",
        cbar_label="Relative Dose Delivered",
        normalize: Literal["max", "center"] | None = None,
    ) -> None:
        arr = self.dose_maps[energy]
        match normalize:  # Normalize to 1 at SSD = 100 cm

            case "center":  # normalize to isocenter
                ref_dose = self.dose_maps[energy][self.y0_idx, self.x0_idx]
                arr = arr / ref_dose
            case "max":  # normalize to the maximum dose
                ref_dose = self.dose_maps[energy].max()
                arr = arr / ref_dose

        im = plt.imshow(arr, cmap="jet", origin="upper", extent=self.extent)
        cbar = plt.colorbar(im)
        cbar.set_label(cbar_label)
        plt.title(title)
        plt.xlabel("Distance (cm)")
        plt.ylabel("Distance (cm)")
        plt.show()

    def plot_PDD(self) -> None:
        """
        Plot PDD along central axis for the current dose map of all 3 energies. Plot starts at depth 0 along the central axis
        """

        # Obtain appropriate dimension y axis. Note coordinate system is plotted to be centered at SSD along central axis.

        depth_upperbound = self.patient_width / 2 + self.ySSD_cm
        for energy, map in self.dose_maps.items():
            pdd = map[self.y0_idx :, self.x0_idx]
            depth = (
                np.arange(pdd.shape[0]) * depth_upperbound / pdd.shape[0]
            )  # create a numpy array from 1 -> (however many pixels deep the patient is), then divide by those pixels (last value now = 1) and scale by the max depth of patient in cm.
            pdd = pdd / pdd.max() * 100  # Normalize
            plt.plot(depth, pdd, label=f"{energy} MeV")
        plt.xlabel("Depth (cm)")
        plt.ylabel("Percent Depth Dose (%)")
        plt.title("PDD Curve")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_lat_dose_profiles(
        self,
        normalization: Literal["total", "individual", "central axis"] = "individual",
    ) -> None:
        """
        Docstring for plot_lat_dose_profiles

        :param normalization: "individual" | "total" | "central axis"
        """
        depth_4cm = int(self.y0_idx + 4 * (self.N / self.patient_width))
        x_min = -patient_width / 2
        x_max = patient_width / 2
        x = np.linspace(x_min, x_max, self.N)
        dose_profile_list = []

        match normalization:
            case "total":
                max_dose = 0
                for energy, map in self.dose_maps.items():
                    dose_profile = map[depth_4cm, :]
                    dose_profile_list.append(dose_profile)
                    max_dose = max(max_dose, dose_profile.max())
                    dose_profile = dose_profile  # Normalize
                for energy, map in self.dose_maps.items():
                    dose_profile = map[depth_4cm, :]
                    plt.plot(x, dose_profile / max_dose * 100, label=f"{energy} MeV")

            case "individual":
                for energy, map in self.dose_maps.items():
                    dose_profile = map[depth_4cm, :]
                    plt.plot(
                        x,
                        dose_profile / dose_profile.max() * 100,
                        label=f"{energy} MeV",
                    )

            case "central axis":
                for energy, map in self.dose_maps.items():
                    dose_profile = map[depth_4cm, :]
                    plt.plot(
                        x,
                        dose_profile / dose_profile[self.x0_idx] * 100,
                        label=f"{energy} MeV",
                    )

        plt.xlabel("Lateral Distance (cm)")
        plt.ylabel("Percent Dose (%)")
        plt.title("Lateral Dose Profile at Depth 4 cm")
        plt.legend()
        plt.grid()
        plt.show()
        pass

    def _create_polar_DPB(self, resolution: dict, energy: float) -> dict:
        """
        Re-create DPB kernel in polar coordinates, only considering the resolution used.
        """
        dtheta = resolution["dtheta"]
        dr = resolution["dr"]
        rmax = resolution["rmax"]

        map = {}  # (r, theta) -> DPB * Existing Dose
        for r in np.arange(0, rmax, dr):
            for theta in range(0, 360, dtheta):

                # DPB at point P' :
                # We want the DPB value an angle theta + 180 (i.e. we want the DPB value at point P due to point P'. Currently, r theta points to point P'.)
                y_pixels = (
                    r
                    * math.sin((theta + 180) * math.pi / 180)
                    / self.kernal_pixel_width
                )
                x_pixels = (
                    r
                    * math.cos((theta + 180) * math.pi / 180)
                    / self.kernal_pixel_width
                )

                y_idx = (self.M - 1) / 2 - y_pixels
                x_idx = (self.M - 1) / 2 + x_pixels

                # Interpolate DPB kernel. Note this is done in indicies, but is the same as if the array was in cm x cm and interpolated a specific cm.
                DPB = map_coordinates(
                    self.DPB_kernals[energy],
                    [[y_idx], [x_idx]],
                    order=1,
                    mode="nearest",
                )[0]

                map[(r, theta)] = DPB
        return map

    def plot_DPB_polar_cartesian(
        self, energy, resolution=None, dpb_polar=None, log10=False
    ) -> None:
        """
        Made with ChatGPT ;-;
        Just for visualization
        """
        if resolution is None:
            resolution = self.resolution
        if dpb_polar is None:
            dpb_polar = self._create_polar_DPB(resolution, energy)

        dr = resolution["dr"]
        rmax = resolution["rmax"]
        dtheta = resolution["dtheta"]

        n = int(np.ceil(2 * rmax / dr)) + 1
        img = np.full((n, n), np.nan)
        center = (n - 1) / 2

        for (r, theta), val in dpb_polar.items():
            theta_rad = math.radians(theta)
            x = r * math.cos(theta_rad)
            y = r * math.sin(theta_rad)
            x_idx = int(round(center + x / dr))
            y_idx = int(round(center + y / dr))
            if 0 <= x_idx < n and 0 <= y_idx < n:
                img[y_idx, x_idx] = val

        z_plot = img
        if log10:
            z_plot = np.log10(np.clip(img, 1e-12, None))
        z_plot = np.ma.masked_invalid(z_plot)

        extent = [-rmax, rmax, -rmax, rmax]
        plt.imshow(z_plot, origin="lower", extent=extent, cmap="jet")
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        plt.title(
            f"DPB Resolution | dr = {dr} cm, dtheta = {dtheta} Deg. | {energy} MeV"
        )
        label = "log10(DPB)" if log10 else "DPB"
        plt.colorbar(label=label)
        plt.axis("equal")
        plt.show()

    def _t(self, P: tuple[int, int], option) -> float:
        """
        Returns the distance traversed by a ray at point P within the patient, in cm.

        If ray trace (diverging ray lines):
        Find distance via minimizing y - mx - b = 0, where y = mx + b describes the ray line towards the point source creating the beam.

        If uniform fluence, the distance travelled through the patient is simply the the vertical distance from point P to the surfarce directly above.

        Parameters:
            P - (x, y) in unit index wrt dose map array
            option (str) – "uniform" | "ray trace".
                Uniform simply takes the vertical path, not taking account for divergence. Used for computation time in taking into account

        """
        x_idx, y_idx = P
        if option == "ray trace":

            # Define new coordinate system in cm, with origin at point source, and x and y swapped (i.e. x increases downwards, and y increases to the right):

            y_cm_from_source = y_cm = self.idx_to_cm(
                x_idx - self.N / 2
            )  # Radial distance
            x_cm = self.idx_to_cm(y_idx)
            x_cm_from_source = (
                self.SSD + x_cm - (self.patient_width / 2 - self.ySSD_cm)
            )  # SPD, i.e. convert from cm from array origin to point source origin

            m = y_cm_from_source / x_cm_from_source  #

            # Find coordinate in surface array that is closest to being in the line y = mx:
            # Test y - mx:
            dist_prev = float("inf")
            for x_idx in range(1, self.surface_arr.shape[0]):
                # Find x and y defined in * rotated coordinate system * with origin point source
                x_int = (
                    self.idx_to_cm(self.surface_arr[x_idx])
                    + self.SSD
                    - (self.patient_width / 2 - self.ySSD_cm)
                )  # Convert to point source coordinate system
                y_int = self.idx_to_cm(x_idx) - self.patient_width / 2
                dist = abs(y_int - m * x_int)
                if dist > dist_prev or x_idx == 511:
                    # Calculate distance between intersection point and point P in cm:
                    dx_cm = x_cm_from_source - x_int
                    dy_cm = y_cm_from_source - y_int
                    t = math.sqrt((dx_cm) ** 2 + (dy_cm) ** 2)
                    return t
                dist_prev = dist
        elif option == "uniform":
            t = self.idx_to_cm(y_idx - self.surface_arr[x_idx])
            return t

    def _read_txt_arr(self, file_name: str) -> np.ndarray:
        arr = np.loadtxt(f"{file_name}.txt", delimiter=" ")
        return arr

    def _save(self, arr: np.ndarray, name: str) -> Path:
        save_dir = Path("Saved Data")
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{name}.npy"
        np.save(path, arr)
        return path

    def _load(self, name: str) -> np.ndarray:
        path = Path("Saved Data") / f"{name}.npy"
        return np.load(path)

    def _x_y_prime_idx(self, r: float, theta: int, x_idx, y_idx) -> tuple[int, int]:
        dx_cm = r * math.cos(theta * math.pi / 180)
        dy_cm = r * math.sin(theta * math.pi / 180)

        dx_idx = round(dx_cm * self.N / self.patient_width)
        dy_idx = round(dy_cm * self.N / self.patient_width)

        x_prime_idx = x_idx + dx_idx
        y_prime_idx = y_idx - dy_idx

        return int(x_prime_idx), int(y_prime_idx)

    def _y_cm(self, y_idx: int) -> float:
        """
        Returns the y position in cm, relative to the dose map coordinate system for a given y idx
        """
        return (self.N / 2 - (y_idx + 1)) * (self.patient_width / self.N)

    def _x_idx_bounds(self, energy: float) -> tuple[int, int]:
        arr = self.dose_maps[energy][
            self.N - 1, :
        ]  # Only examine the last row (since we know beam diverges, and will be largest width)
        x_min = x_max = None
        for i in range(1, self.N):
            if arr[i - 1] == 0 and arr[i] != 0:
                x_min = i
            elif arr[i] == 0 and arr[i - 1] != 0:
                x_max = i - 1
        return x_min, x_max

    def _DPB_polar(self, r: float, theta: int, energy: float) -> float:

        x_cm = r * math.cos(theta * math.pi / 180)
        y_cm = r * math.sin(theta * math.pi / 180)

        x_idx = (x_cm / self.kernal_pixel_width) + (self.M - 1) / 2
        y_idx = (self.M - 1) / 2 - (y_cm / self.kernal_pixel_width)

        return self.DPB_kernals[energy][(int(y_idx), int(x_idx))]

    def _find_surface_arr(self) -> np.ndarray:
        surface_arr = np.empty(self.N)
        for x_idx in range(self.N):
            for y_idx in range(1, self.N):
                if (
                    self.patient_mask[y_idx, x_idx]
                    != self.patient_mask[y_idx - 1, x_idx]
                ):
                    surface_arr[x_idx] = y_idx
                    break
        return surface_arr
        # Find 1D array describing surface y coordinate (idx) as a function of x (idx)

    def idx_to_cm(self, idx: int) -> float:
        cm = (idx + 1) * (self.patient_width / self.N)
        return cm

    def cm_to_idx(self, cm: float) -> int:
        return cm * (self.N / self.patient_width) - 1

    def _test_plot(
        self,
    ) -> None:
        xs = []
        ys = []
        for theta in range(0, 360, 30):
            for r in np.arange(0, 4, 0.1):
                x = r * math.cos(theta * math.pi / 180)
                y = r * math.sin(theta * math.pi / 180)
                xs.append(x)
                ys.append(y)
            print(theta)
        plt.scatter(xs, ys)
        plt.title = "Polar Resolution Visualization"
        plt.show()

    def _test_plot_patient_surface(self, unit: str = "cm") -> None:
        arr = self.patient_mask
        if unit == "cm":
            im = plt.imshow(arr, cmap="binary", origin="upper", extent=self.extent)
        elif unit == "idx":
            im = plt.imshow(arr, cmap="binary", origin="upper")
        plt.title("2D Patient Surface")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Distance (cm)")
        plt.grid()
        plt.show()

    def _test_plot_fluence_no_mask(self) -> None:
        arr = self.fluence_wo_mask

        im = plt.imshow(arr, cmap="jet", origin="upper", extent=self.extent)
        plt.title("2D Patient Surface")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Distance (cm)")
        plt.grid()
        plt.show()


# –––––––––––––––––– Execution ––––––––––––––––––
linear_attenuation_coeff = {1.25: 0.0632, 10: 0.0222, 20: 0.0182}  # MeV: cm^-1

SSD = 100  # cm
beam_width0 = 6  # cm
patient_width = 15  # cm
kernal_pixel_width = 0.1  # cm
resolution = {"dr": 0.1, "dtheta": 10, "rmax": 8}

patient = DoseMap(
    SSD=SSD,
    beam_width0=beam_width0,
    patient_width=patient_width,
    kernal_pixel_width=kernal_pixel_width,
    mu=linear_attenuation_coeff,
    resolution=resolution,
)

"""
Note energy must be 1.25 | 10 | 20 type float. Code is not modular for other energies. 
"""


# –––––––––––––––––––––––––––––-- Visualization –––––––––––––––––––––––––––––--
# patient._test_plot_patient_surface(unit="idx")
# patient._test_plot_patient_surface()
# patient.create_fluence(energy)
# patient._test_plot_fluence_no_mask()
# patient.apply_attenuation(energy, option="uniform")
# patient.plot_DPB_polar_cartesian(energy, log10=True)
# patient.apply_DPB_kernel(energy)
# patient.plot_dose_map(energy, title=f"{energy} MeV | 2D Dose with Applied DPB Kernel")

# ––––––– Deliverable 1: Display each kernel with appropriate scale/cmap –––––––
patient.plot_DPB_kernal(1.25)
patient.plot_DPB_kernal(10)
patient.plot_DPB_kernal(20)

# ––––––– Deliverable 2: Display primary fluence wo phantom (i.e. ISL + beam width) –––––––
patient.create_fluence(energy=1.25)
patient.create_fluence(energy=10)
patient.create_fluence(energy=20)
patient.plot_dose_map(
    energy=1.25,
    title="Primary Fluence in Absence of Phantom",
    cbar_label="Relative Fluence",
)

# ––––––– Deliverable 3: Display attenuation in tissue along ray-lines –––––––
patient.apply_attenuation(energy=1.25, option="uniform")
patient.apply_attenuation(energy=10, option="uniform")
patient.apply_attenuation(energy=20, option="uniform")
patient.plot_dose_map(
    energy=1.25, title=f"{1.25} MeV 2D Dose Map | Attenuation in Tissue"
)
patient.plot_dose_map(energy=10, title=f"{10} MeV 2D Dose Map | Attenuation in Tissue")
patient.plot_dose_map(energy=20, title=f"{20} MeV 2D Dose Map | Attenuation in Tissue")

# ––––––– Deliverable 4: Dose distribution –––––––
# patient.apply_DPB_conv(1.25)
# patient.apply_DPB_conv(10)
# patient.apply_DPB_conv(20)
# patient.plot_dose_map(1.25, title=f"{1.25} MeV 2D Dose Plot via Convolution for Troubleshooting")
# patient.plot_dose_map(10, title=f"{10} MeV 2D Dose Plot via Convolution for Troubleshooting")
# patient.plot_dose_map(20, title=f"{20} MeV 2D Dose Plot via Convolution for Troubleshooting")

# patient.apply_DPB_kernel(energy=1.25)
# patient.apply_DPB_kernel(energy=10)
# patient.apply_DPB_kernel(energy=20)
# patient._save(patient.dose_maps[1.25], "1_25_MeV_Dose_Map")
# patient._save(patient.dose_maps[10], "10_MeV_Dose_Map")
# patient._save(patient.dose_maps[20], "20_MeV_Dose_Map")

patient.dose_maps[1.25] = patient._load("1_25_MeV_Dose_Map")
patient.dose_maps[10] = patient._load("10_MeV_Dose_Map")
patient.dose_maps[20] = patient._load("20_MeV_Dose_Map")

patient.plot_dose_map(
    1.25, title=f"{1.25} MeV | 2D Dose with Applied DPB Kernel", normalize="max"
)
patient.plot_dose_map(
    10, title=f"{10} MeV | 2D Dose with Applied DPB Kernel", normalize="max"
)
patient.plot_dose_map(
    20, title=f"{20} MeV | 2D Dose with Applied DPB Kernel", normalize="max"
)

# ––––––– Deliverable 5: PDD curves –––––––
patient.plot_PDD()

# ––––––– Deliverable 6:  curves –––––––
patient.plot_lat_dose_profiles(normalization="central axis")
