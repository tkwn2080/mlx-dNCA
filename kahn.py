import numpy as np
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy import ndimage
import random

class MandelbrotLocationFinder:
    def __init__(self, max_iterations=100):
        self.max_iterations = max_iterations

    def calculate_mandelbrot(self, xmin, xmax, ymin, ymax, width, height):
        """Calculate Mandelbrot set for given bounds"""
        x = np.linspace(xmin, xmax, width)
        y = np.linspace(ymin, ymax, height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y

        Z = np.zeros_like(C)
        M = np.zeros(C.shape, dtype=int)

        for i in range(self.max_iterations):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C[mask]
            M[mask] = i

        return M

    def calculate_entropy(self, image):
        """Calculate Shannon entropy of an image to measure complexity"""
        # Normalize to 0-255 range
        normalized = ((image / self.max_iterations) * 255).astype(np.uint8)

        # Calculate histogram
        hist, _ = np.histogram(normalized.flatten(), bins=256, range=(0, 255))
        hist = hist[hist > 0]  # Remove zero entries

        # Calculate probabilities
        prob = hist / hist.sum()

        # Calculate entropy
        entropy = -np.sum(prob * np.log2(prob))

        return entropy

    def calculate_edge_density(self, image):
        """Calculate edge density as another measure of interest"""
        # Apply Sobel edge detection
        edges = ndimage.sobel(image.astype(float))

        # Calculate percentage of pixels that are edges
        threshold = np.percentile(np.abs(edges), 80)
        edge_pixels = np.sum(np.abs(edges) > threshold)
        total_pixels = edges.size

        return edge_pixels / total_pixels

    def is_interesting(self, image, min_entropy=4.0, min_edge_density=0.1):
        """Determine if a location is interesting enough"""
        entropy = self.calculate_entropy(image)
        edge_density = self.calculate_edge_density(image)

        # Check if not all black or all colored (boring areas)
        unique_values = len(np.unique(image))
        if unique_values < 5:
            return False, 0

        # Calculate overall interest score
        interest_score = (entropy / 8.0) * 0.6 + edge_density * 0.4

        return entropy >= min_entropy and edge_density >= min_edge_density, interest_score

    def find_interesting_locations(self, num_locations=50,
                                 search_attempts=1000,
                                 resolution=(512, 512)):
        """Find interesting locations in the Mandelbrot set"""
        locations = []
        width, height = resolution

        # Define search regions with different zoom levels
        search_regions = [
            # Full view
            {"xmin": -2.5, "xmax": 1.0, "ymin": -1.25, "ymax": 1.25, "zoom": 1},
            # Main bulb area
            {"xmin": -0.8, "xmax": -0.4, "ymin": -0.2, "ymax": 0.2, "zoom": 5},
            # Seahorse valley
            {"xmin": -0.75, "xmax": -0.73, "ymin": 0.09, "ymax": 0.11, "zoom": 50},
            # Elephant valley
            {"xmin": 0.275, "xmax": 0.280, "ymin": 0.006, "ymax": 0.010, "zoom": 200},
            # Deep zoom areas
            {"xmin": -0.7533, "xmax": -0.7532, "ymin": 0.1138, "ymax": 0.1139, "zoom": 10000},
        ]

        attempts = 0
        while len(locations) < num_locations and attempts < search_attempts:
            # Choose a random search region
            region = random.choice(search_regions)

            # Random location within the region
            x_center = random.uniform(region["xmin"], region["xmax"])
            y_center = random.uniform(region["ymin"], region["ymax"])

            # Random zoom level variation
            zoom = region["zoom"] * random.uniform(0.5, 2.0)

            # Calculate bounds
            x_range = 3.5 / zoom
            y_range = 2.5 / zoom
            xmin = x_center - x_range/2
            xmax = x_center + x_range/2
            ymin = y_center - y_range/2
            ymax = y_center + y_range/2

            # Calculate Mandelbrot
            mandelbrot = self.calculate_mandelbrot(xmin, xmax, ymin, ymax, width, height)

            # Check if interesting
            is_good, score = self.is_interesting(mandelbrot)

            if is_good:
                location = {
                    "x": x_center,
                    "y": y_center,
                    "zoom": zoom,
                    "score": score,
                    "difficulty": self._calculate_difficulty(zoom)
                }
                locations.append(location)
                print(f"Found location {len(locations)}/{num_locations} - "
                      f"Score: {score:.3f}, Zoom: {zoom:.1f}")

            attempts += 1

        # Sort by difficulty
        locations.sort(key=lambda x: x["difficulty"])

        return locations

    def _calculate_difficulty(self, zoom):
        """Calculate difficulty based on zoom level"""
        if zoom < 2:
            return 1  # Easy - full view
        elif zoom < 10:
            return 2  # Medium - slightly zoomed
        elif zoom < 100:
            return 3  # Hard - significantly zoomed
        elif zoom < 1000:
            return 4  # Very Hard - deep zoom
        else:
            return 5  # Extreme - ultra deep zoom

    def save_locations(self, locations, filename="mandelbrot_locations.json"):
        """Save locations to JSON file"""
        with open(filename, 'w') as f:
            json.dump(locations, f, indent=2)
        print(f"Saved {len(locations)} locations to {filename}")

    def visualize_location(self, location, resolution=(512, 512)):
        """Visualize a single location"""
        width, height = resolution
        x_range = 3.5 / location["zoom"]
        y_range = 2.5 / location["zoom"]

        xmin = location["x"] - x_range/2
        xmax = location["x"] + x_range/2
        ymin = location["y"] - y_range/2
        ymax = location["y"] + y_range/2

        mandelbrot = self.calculate_mandelbrot(xmin, xmax, ymin, ymax, width, height)

        plt.figure(figsize=(8, 6))
        plt.imshow(mandelbrot, cmap='hot', extent=[xmin, xmax, ymin, ymax])
        plt.colorbar(label='Iterations to divergence')
        plt.title(f'Location: ({location["x"]:.6f}, {location["y"]:.6f}), '
                  f'Zoom: {location["zoom"]:.1f}x')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.show()

def main():
    # Create finder instance
    finder = MandelbrotLocationFinder(max_iterations=100)

    # Find interesting locations
    print("Searching for interesting Mandelbrot locations...")
    locations = finder.find_interesting_locations(num_locations=100)

    # Save to file
    finder.save_locations(locations)

    # Visualize a few examples
    print("\nVisualizing sample locations...")
    for i in range(min(3, len(locations))):
        print(f"\nLocation {i+1}:")
        print(f"  Coordinates: ({locations[i]['x']:.6f}, {locations[i]['y']:.6f})")
        print(f"  Zoom: {locations[i]['zoom']:.1f}x")
        print(f"  Difficulty: {locations[i]['difficulty']}")
        print(f"  Score: {locations[i]['score']:.3f}")
        finder.visualize_location(locations[i])

if __name__ == "__main__":
    main()
