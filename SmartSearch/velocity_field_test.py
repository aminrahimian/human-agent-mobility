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

plt.figure(figsize=(8,6))
plt.scatter(px, py, c=pv, cmap='viridis', s=5, alpha=0.5)
plt.colorbar()
plt.title('Original Flow Field from Particle Data')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.savefig('original_flow_field.png')  # Save the figure
plt.show()

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
from scipy.ndimage import gaussian_filter  # Import Gaussian filter

# Create a new grid to interpolate onto
grid_x, grid_y = np.mgrid[0:1600:16, 0:1600:16]  # 100x100 grid
# Interpolate the velocity components onto the new grid
grid_vx = griddata((px,py), pvx, (grid_x, grid_y), method='linear')
grid_vy = griddata((px,py), pvy, (grid_x, grid_y), method='linear')
grid_v = np.sqrt(grid_vx**2 + grid_vy**2)
# Replace NaN values with 0 (correct syntax)
grid_vx = np.nan_to_num(grid_vx, nan=0.0)  # Fills NaNs with 0
grid_vy = np.nan_to_num(grid_vy, nan=0.0)
grid_v = np.nan_to_num(grid_v, nan=0.0)  # Apply to magnitude as well

interp_fn = RegularGridInterpolator((grid_x[:,0], grid_y[0,:]), grid_vx, method='linear')
# Query multiple points
points = np.array([800.5, 400.3])  # Shape (N, 2)
values = interp_fn(points)  # Returns array of interpolated values
print(values)

# Plot interpolated velocity magnitude
vmax = np.nanmax(grid_v)  # Determine the maximum value for consistent colorbar
vmin = np.nanmin(grid_v)  # Determine the minimum value for consistent colorbar

plt.figure(figsize=(8, 6))
im = plt.imshow(grid_v.T, extent=(0, 1600, 0, 1600), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)  # Use vmin and vmax
plt.title('Interpolated Velocity Magnitude')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
# plt.scatter(grid_x, grid_y, c=grid_v, cmap='viridis', s=5, alpha=1)
plt.colorbar(im)
plt.savefig('interpolated_velocity_magnitude.png')  # Save the figure
plt.show()

# Apply the Gaussian filter
sigma = 3.0  # Adjust this to control the smoothing.  The standard deviation for Gaussian kernel. Typical values are 1-10
filtered_grid_vx = gaussian_filter(grid_vx, sigma=sigma)
filtered_grid_vy = gaussian_filter(grid_vy, sigma=sigma)
filtered_grid_v = np.sqrt(filtered_grid_vx**2 + filtered_grid_vy**2)
plt.figure(figsize=(8,6))
im = plt.imshow(filtered_grid_v.T, extent=(0, 1600, 0, 1600), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)  # Use consistent vmin and vmax
plt.title(f'Filtered Velocity Magnitude (Gaussian, sigma={sigma})')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.colorbar(im)
plt.savefig(f'gaussian_filtered_velocity_magnitude_sigma_{sigma}.png')  # Save the figure
plt.show()