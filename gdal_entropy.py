from osgeo import gdal
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import PercentFormatter
import numpy as np
import math
import matplotlib.pyplot as pyplot

# ---- Get GDAL Elevation Data ----
FILEPATH = 'n00_e010_1arc_v3.tif'  # Gabon Filepath
gdal.UseExceptions()

ds = gdal.Open(FILEPATH)
band = ds.GetRasterBand(1)
elevation = band.ReadAsArray()  # returns a nd-array with dimensions equal to .tif dims
geo_transform = ds.GetGeoTransform()

# ---- GLOBAL VARIABLES ----
CHUNK_SIZE = 64
INCREMENT = 2
DIMENSIONS = (40, 40)
CHUNK_SHAPE = CHUNK_SIZE/INCREMENT
CHUNK_OFFSET = (0,0)



# ---- MAIN CLASS ----
class Chunk:
    def __init__(self, size, offset, increment):
        self.size: int = size
        self.offset: tuple = offset
        self.increment: int = increment
        self.location = self.chunk_location()
        self.matrix = self.chunk_matrix()
        self.points = self.chunk_points()  # In form north, east, altitude of floats
        self.vector_coefficients = self.chunk_vector()
        self.displacement_vector = self.displacement_per_point()

        # Ideal mu is 0 because ideal distribution will be perfect.
        # This is acceptable because it is not finding z-score for itself, but against
        # the average chunk across the world
        self.sigma = 30  # Approximation for std against chunk size at large values
        self.z_score_vector = self.z_score_per_point()
        self.entropy = self.chunk_entropy()/self.size ** 2

    def chunk_location(self) -> object:
        global elevation, geo_transform

        """North_min = North_base + (offset * pixel incremement)"""
        """North_max = North_min + (CHUNK SIZE + pixel increment) = North_base = pixel increment * (size + offset)"""
        location: object = [geo_transform[0] + (self.offset[0] * geo_transform[1]),
                            geo_transform[0] + geo_transform[1] * (self.size + self.offset[0]),
                            geo_transform[3] + (self.offset[1] * geo_transform[5]),
                            geo_transform[3] + geo_transform[5] * (self.size + self.offset[1])]
        return location

    def chunk_matrix(self):
        full_matrix = band.ReadAsArray(self.offset[0], self.offset[1], self.size, self.size)
        chunk_matrix = full_matrix[::self.increment, ::self.increment]
        return chunk_matrix

    def chunk_points(self):
        global CHUNK_SHAPE, geo_transform
        points = [[], [], []]
        north, east = 0, 0
        while north < CHUNK_SHAPE:
            while east < CHUNK_SHAPE:
                points[0].append(self.location[0] + (east * self.increment * geo_transform[1]))
                points[1].append(self.location[2] + (north * self.increment * geo_transform[5]))
                points[2].append(self.matrix[north][east])
                east += 1
            north += 1
            east = 0

        return np.array(points)

    def chunk_vector(self):
        tmp_A = []
        tmp_b = []
        xs, ys, zs = self.points
        for i in range(len(xs)):
            tmp_A.append([xs[i], ys[i], 1])
            tmp_b.append(zs[i])
        b = np.matrix(tmp_b).T
        A = np.matrix(tmp_A)

        # Matrix Algebra of Ax=b to solve for x
        # In form Inverse(A * A-Transpose) * A-Transpose * b
        fit = (A.T * A).I * A.T * b
        return fit

    def displacement_per_point(self):
        displacement = []
        fit = np.asarray(self.vector_coefficients).ravel()
        for i, elem in enumerate(self.points[2]):
            X = self.points[0][i]
            Y = self.points[1][i]
            expected_val = fit[0] * X + fit[1] * Y + fit[2]
            displacement.append(elem - expected_val)

        return displacement

    def percentage_fit(self):
        differences = []
        fit = np.asarray(self.vector_coefficients).ravel()
        for i, elem in enumerate(self.points[2]):
            X = self.points[0][i]
            Y = self.points[1][i]
            expected_val = fit[0] * X + fit[1] * Y + fit[2]
            differences.append(1- abs((elem/expected_val)-1))

        return differences

    def z_score_per_point(self):
        z_scores = []

        for point in self.displacement_vector:
            z_score = abs(point/self.sigma)
            z_scores.append(z_score)

        return z_scores

    def chunk_entropy(self):
        entropy = 0
        for score in self.z_score_vector:
            P_ak = 1/(4 ** score)  # Arbitrary, produces reasonable probability values
            entropy += -math.log(P_ak, 2)  # Claude Shannon's formula for entropy of a system

        adjusted_entropy = entropy * (self.increment ** 2)  # Adjusts for chunk increment
        return adjusted_entropy


# ---- PLOTTING ----
class Plot(Chunk):
    def __init__(self, size, offset, increment):
        super().__init__(size, offset, increment)

    def plot_raster(self):
        x0, x1, y0, y1 = self.location
        plt.imshow(self.matrix, cmap='gist_earth', extent=[x0, x1, y1, y0])
        plt.colorbar()
        plt.show()

    def plot_mesh_3d(self):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')

        x0, x1, y0, y1 = self.location
        X = np.arange(x0, x1, geo_transform[1] * INCREMENT)
        Y = np.arange(y0, y1, geo_transform[5] * INCREMENT)
        X, Y = X[:int(CHUNK_SHAPE)], Y[:int(CHUNK_SHAPE)] # Accounting for over-counting due to float imprecision

        X, Y = np.meshgrid(X, Y)
        Z = self.matrix

        ax.plot_surface(X, Y, Z, cmap=cm.terrain, linewidth=0, antialiased=False)

        fit = np.asarray(self.vector_coefficients)

        Z = fit[0] * X + fit[1] * Y + fit[2]
        ax.plot_wireframe(X, Y, Z, color='k', alpha=0.5)

        ax.text2D(0.01, 0.99, f"Information (bits): {round(chunk.entropy, 4)}", transform=ax.transAxes)
        ax.text2D(0.01, 0.93, f"Latitude: {round(chunk.location[0], 5)}, {round(chunk.location[1], 5)}", transform=ax.transAxes)
        ax.text2D(0.01, 0.87, f"Longitude: {360 - round(chunk.location[2], 5)}, {360 - round(chunk.location[3], 5)}", transform=ax.transAxes)
        ax.set_xlabel('East')
        ax.set_ylabel('North')
        ax.set_zlabel('Altitude (m)')
        plt.show()

    def displacement_histogram(self, histogram_of):
        mu = np.mean(histogram_of)
        count, bins, ignored = plt.hist(histogram_of, 50, weights=np.ones(len(histogram_of)) / len(histogram_of))
        plt.plot(bins, 1 / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * self.sigma ** 2)), linewidth=2, color='r')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        pyplot.xlabel('Z-Score')

        plt.show()


#---- MAIN ----
if __name__ == "__main__":
    east, north = 0, 0
    diffs = []
    while north < DIMENSIONS[0]:
        while east < DIMENSIONS[1]:
            CHUNK_OFFSET = (north * CHUNK_SIZE, east * CHUNK_SIZE)
            chunk = Chunk(CHUNK_SIZE, CHUNK_OFFSET, INCREMENT)
            #plot = Plot(CHUNK_SIZE, CHUNK_OFFSET, INCREMENT)
            #raster_plot = plot.plot_raster()
            #mesh_plot = plot.plot_mesh_3d()
            #plot.displacement_histogram(chunk.chunk_probabilities)
            differences = chunk.percentage_fit()

            diffs += differences
            east += 1
        north += 1
        east = 0

    #print(diffs)
    print(np.mean(diffs))



