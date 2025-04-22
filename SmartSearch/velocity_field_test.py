import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


mainfolder = os.getcwd()
# Load wind data from .mat files
def plot_vel_vector(directory=mainfolder+'\\OneDrive_1_8-7-2024', frame_range=(7, 7)):
    if not os.path.exists(directory):
        print(f"Directory {directory} not found, using synthetic wind data.")
        x = np.linspace(0, 1600, 20)
        y = np.linspace(0, 1600, 20)
        px, py = np.meshgrid(x, y)
        pvx = np.sin(px / 1600 * 2 * np.pi) * 2
        pvy = np.cos(py / 1600 * 2 * np.pi) * 2
        return px.flatten(), py.flatten(), pvx.flatten(), pvy.flatten()

    px, py, pvx, pvy = [], [], [], []
    for frame in range(frame_range[0], frame_range[1] + 1):
        # Add zero-padding to frame number (e.g., 006.mat)
        mat_file = os.path.join(directory, f"{frame:05d}.mat")
        if not os.path.exists(mat_file):
            print(f"Warning: {mat_file} not found, skipping.")
            continue
        try:
            data = loadmat(mat_file)
            p = data['p']  # Shape: (2, N)
            px.append(p[0, :])
            py.append(p[1, :])
            pvx.append(data['pvx'].flatten())
            pvy.append(data['pvy'].flatten())
        except Exception as e:
            print(f"Error reading {mat_file}: {e}")
            continue
    if not px:  # If no files were loaded
        raise FileNotFoundError("No valid .mat files found in directory.")
    px = np.concatenate(px)
    py = np.concatenate(py)
    pvx = np.concatenate(pvx)
    pvy = np.concatenate(pvy)
    return px, py, pvx, pvy

[px, py, pvx, pvy] = plot_vel_vector()
print(f"px shape: {px.shape}, py shape: {py.shape}, pvx shape: {pvx.shape}, pvy shape: {pvy.shape}")

# Plot PDF of pvx, pvy, and pv
pv = np.sqrt(pvx**2 + pvy**2)
# Specify number of bins
num_bins = 50  # Change this to your desired number of bins

# plt.figure(figsize=(8,6))
# plt.scatter(px, py, c=pv, cmap='viridis', s=5, alpha=0.5)
# plt.colorbar()
# plt.show()

# # Plot PDFs using histograms
# plt.figure(figsize=(8, 6))
# for data, label, color in [(pvx, 'pvx', 'blue'), (pvy, 'pvy', 'green'), (pv, 'pv', 'red')]:
#     # Compute histogram with density
#     counts, bins = np.histogram(data, bins=num_bins, density=True)
#     # Calculate bin centers for smooth curve
#     bin_centers = (bins[:-1] + bins[1:]) / 2
#     # Plot as a smooth curve by connecting bin centers
#     plt.plot(bin_centers, counts, label=label, color=color, linewidth=2)
# plt.title(f'PDFs of pvx, pvy, and pv ({num_bins} bins)')
# plt.xlabel('Velocity')
# plt.ylabel('Density')
# plt.legend()
# plt.grid(True)
# plt.show()

# ========================================= interpolation =========================================
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator

# Create a new grid to interpolate onto
grid_x, grid_y = np.mgrid[0:1600:16, 0:1600:16]  # 100x100 grid
# Interpolate the velocity components onto the new grid
grid_vx = griddata((px,py), pvx, (grid_x, grid_y), method='linear')
grid_vy = griddata((px,py), pvy, (grid_x, grid_y), method='linear')
grid_v = np.sqrt(grid_vx**2 + grid_vy**2)
# Replace NaN values with 0 (correct syntax)
grid_vx = np.nan_to_num(grid_vx, nan=0.0)  # Fills NaNs with 0
grid_vy = np.nan_to_num(grid_vy, nan=0.0)

interp_fn = RegularGridInterpolator((grid_x[:,0], grid_y[0,:]), grid_vx, method='linear')
# Query multiple points
points = np.array([800.5, 400.3])  # Shape (N, 2)
values = interp_fn(points)  # Returns array of interpolated values
print(values)
# plt.figure(figsize=(8,6))
# # plt.imshow(grid_v.T, extent=(0, 1, 0, 1), origin='lower')
# plt.scatter(grid_x, grid_y, c=grid_v, cmap='viridis', s=5)
# plt.colorbar()
# plt.show()