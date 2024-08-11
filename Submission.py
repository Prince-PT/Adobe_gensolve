import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def smooth_path(path, window_length=5, polyorder=2):
    smoothed_path = []
    for segment in path:
        x = segment[:, 0]
        y = segment[:, 1]
        smoothed_x = savgol_filter(x, window_length, polyorder)
        smoothed_y = savgol_filter(y, window_length, polyorder)
        smoothed_path.append(np.column_stack((smoothed_x, smoothed_y)))
    return smoothed_path

def extract_features(path):

    points = np.vstack(path)
    hull = ConvexHull(points)
    area = hull.volume
    perimeter = hull.area
    convexity = perimeter / np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    aspect_ratio = np.ptp(points[:, 0]) / np.ptp(points[:, 1])
    
    # Fourier Descriptors (simplified)
    fourier_transform = np.fft.fft(points[:, 0] + 1j * points[:, 1])
    fourier_descriptor = np.abs(fourier_transform[:5])  # Take first few descriptors

    return {
        "area": area,
        "perimeter": perimeter,
        "convexity": convexity,
        "aspect_ratio": aspect_ratio,
        "fourier_descriptor": fourier_descriptor,
    }

def train_shape_classifier(paths):
    
    feature_list = [extract_features(path) for path in paths]
    X = np.array([list(f.values())[:-1] for f in feature_list])  # Exclude Fourier Descriptors for clustering

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_pca)

    return kmeans, pca, scaler

def classify_shape(path, model, pca, scaler):
    features = extract_features(path)
    X = np.array([list(features.values())[:-1]])  # Exclude Fourier Descriptors

    # Standardize and transform features
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    # Use model to predict shape class
    shape_class = model.predict(X_pca)
    
    # Map cluster labels to shape descriptions
    shape_labels = {0: "Circle", 1: "Square/Rectangle", 2: "Star"}
    
    return shape_labels.get(shape_class[0], "Unknown Shape")


def classify_doodles(csv_path):
    doodles = read_csv(csv_path)
    smoothed_doodles = [smooth_path(doodle) for doodle in doodles]
    
    # Train model
    model, pca, scaler = train_shape_classifier(smoothed_doodles)
    
    for i, doodle in enumerate(smoothed_doodles):
        shape = classify_shape(doodle, model=model, pca=pca, scaler=scaler)
        print(f"Doodle {i + 1}: {shape}")

def reflect_point(point, axis, center=None):
    if axis == 'vertical':
        return np.array([-point[0], point[1]])
    elif axis == 'horizontal':
        return np.array([point[0], -point[1]])
    elif axis == 'diagonal':
        if center is None:
            raise ValueError("Center must be provided for diagonal reflection")
        return np.array([2 * center[0] - point[0], 2 * center[1] - point[1]])
    else:
        raise ValueError("Unsupported axis")

def is_circle(path):
    # Check if the shape is a circle based on aspect ratio or convex hull
    all_points = np.vstack(path)
    centroid = np.mean(all_points, axis=0)
    distances = np.linalg.norm(all_points - centroid, axis=1)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    return std_distance < 1e-6  # Small tolerance for circularity

def has_symmetry(path):
    def check_axis_symmetry(axis):
        if axis == 'diagonal':
            all_points = np.vstack(path)
            center = np.mean(all_points, axis=0)
        else:
            center = None
        
        for segment in path:
            points = np.array(segment)
            reflected_points = np.array([reflect_point(p, axis, center) for p in points])
            points = np.round(points, decimals=8)
            reflected_points = np.round(reflected_points, decimals=8)
            
            for p in points:
                if not any(np.all(np.isclose(reflected_points, rp, atol=1e-6), axis=1).any() for rp in reflected_points):
                    return False
        return True

    if is_circle(path):
        return float('inf')  # Infinite lines of symmetry for a circle

    symmetries = 0
    if check_axis_symmetry('vertical'):
        symmetries += 1
    if check_axis_symmetry('horizontal'):
        symmetries += 1
    if check_axis_symmetry('diagonal'):
        symmetries += 1
    
    return symmetries

def classify_doodles_and_check_symmetry(csv_path):
    doodles = read_csv(csv_path)
    smoothed_doodles = [smooth_path(doodle) for doodle in doodles]
    
    # Train model
    model, pca, scaler = train_shape_classifier(smoothed_doodles)
    
    for i, doodle in enumerate(smoothed_doodles):
        shape = classify_shape(doodle, model=model, pca=pca, scaler=scaler)
        print(f"Doodle {i + 1}: {shape}")

        # Check symmetry
        symmetry_count = has_symmetry(doodle)
        if symmetry_count == float('inf'):
            print(f"Doodle {i + 1} is a circle with infinite lines of symmetry.")
        elif symmetry_count > 0:
            print(f"Doodle {i + 1} has {symmetry_count} axis(es) of symmetry.")
        else:
            print(f"Doodle {i + 1} has no axes of symmetry.")

# Classify and check symmetry of doodles in the provided CSV
csv_path = 'problems/isolated.csv'
classify_doodles_and_check_symmetry(csv_path)
