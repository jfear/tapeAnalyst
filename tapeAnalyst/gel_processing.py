#!/usr/bin/env python
import numpy as np
from skimage import io, color
from scipy import signal as signal
import pickle

class TapeStationGel():
    """ Class for handling Tape Station Gels.
    
    :Args:
        :param fname: Name of gel image file [PNG].
        :type fname: str

        :param ladder: Location of the ladder if present.
        :type ladder: int
    
    :Attr:
        :param self.colorGel: Representation of original color gel as a 3d matrix.
        :type self.colorGel: numpy.ndarray

        :param self.grayGel: gray scale representation of gel as a 2d matrix.
        :type self.grayGel: numpy.ndarray

        :param self.ladder: Location of ladder if present
        :type self.ladder: int

        :param self.gelTop: Location of the top of the gel (Purple band).
        :type self.gelTop: int

        :param self.gelBottom: Location of the bottom of the gel (Green band).
        :type self.gelBottom: int

        :param self.gelLeft: Location of the left of the gel (Green and Purple band).
        :type self.gelLeft: int

        :param self.gelRigth: Location of the right of the gel (Green and Purple band).
        :type self.gelRigth: int

        :param self.laneLocations: A list of tuples containing (start, end,
            width, midpoint) of each lane.
        :type self.laneLocations: list of tuples

    """
    def __init__(self, fname, sample):
        # read gel image
        self.colorGel = io.imread(fname)

        # Convert gel to gray scale for intensity analysis
        self.grayGel = 1 - color.rgb2gray(self.colorGel)

        # Figure out the top and bottom coordinates of gel using the dye
        self.getGelRegion()

        # Parse lanes using dye front and end
        self.splitLanes(sample)

    def getGelRegion(self):
        """ Identify the region in the image that contains the gel.

        Using the dye front (Green) and the dye end (Purple) figure out the
        coordinates in the image that contain the gel. Select the regions with
        "green" or "purple" colors. 

        :Attr:
            :param self.gelBottom: The lowest y-axis location of the dye front.
            :type self.gelBottom: int
            
            :param self.gelTop: The highest y-axis location of the dye end.
            :type self.gelTop: int

            :param self.gelLeft: The lowest x-axis location of the dyes (front
                or end).
            :type self.gelLeft: int

            :param self.gelRight: The highest x-axis location of the dyes (front
                or end).
            :type self.gelRight: int

            :param self.dyeGel: Gel image containing only the dye front and dye
                end.
            :type self.dyeGel: numpy.ndarray

        """
        # Get bottom coords from dye front (GREEN).
        dyeFrontGel = self.dyeMarker(dye='front')
        dyeFrontCoords = np.nonzero(dyeFrontGel.sum(axis=1))[0]
        if dyeFrontCoords[-1].any():
            self.gelBottom = dyeFrontCoords[-1]
        else:
            self.gelBottom = None

        # Get top coords from dye end (PURPLE)
        dyeEndGel = self.dyeMarker(dye='end')
        dyeEndCoords = np.nonzero(dyeEndGel.sum(axis=1))[0]
        if dyeEndCoords[0].any():
            self.gelTop = dyeEndCoords[0]
        else:
            self.gelTop = None

        # Get left and right coords by comibining green and purple
        self.dyeGel = dyeFrontGel + dyeEndGel
        dyeCoords = np.nonzero(self.dyeGel.sum(axis=0))[0]
        self.gelLeft = dyeCoords[0]
        self.gelRight = dyeCoords[-1]

    def dyeMarker(self, dye='front'):
        """ Set boolean mask on gel image to identify dye front or dye end. 

        The TapeStation annotates the dye front as green and the dye end as purple.
        Using these color profiles, mask the remainder of the image to be
        white, effectively selecting the regions with these colors.

        :Args:
            :param dye: Name of the dye you want to locate {'front', 'end'}
            :type dye: str
            
        :returns: Masked array

        """
        if dye == 'front':
            # Green
            maskR = self.colorGel[:, :, 0] < 10
            maskG = self.colorGel[:, :, 1] > 10
            maskB = self.colorGel[:, :, 3] > 10
        elif dye == 'end':
            # Purple
            maskR = self.colorGel[:, :, 0] > 10
            maskG = self.colorGel[:, :, 1] < 10
            maskB = self.colorGel[:, :, 3] > 10
        else:
            print("dye must be 'front' or 'end'")
            raise ValueError
        
        # Set everything that is not in my mask to 0
        image2 = self.colorGel.copy()
        image2[~maskR] = 0
        image2[~maskG] = 0
        image2[~maskB] = 0

        return color.rgb2gray(image2)

    def splitLanes(self, sample):
        """ Using the dye front and end, separate lanes.

        """
        # Create row vector where lanes are values above 0
        dyeLanes = np.nonzero(self.dyeGel.sum(axis=0))[0]

        # Group consecutive pixels that are spearated by 0's into lanes
        laneLocations = self.getLaneLocations(dyeLanes)

        # Iterate of lanes and create Lane objects
        lanes = list()
        for index, lane in enumerate(laneLocations):
            row = sample.iloc[index]
            lanes.append(GelLane(gel=self.grayGel, wellID=row['wellID'],
                                 description=row['description'], **lane))

    def getLaneLocations(self, arr):
        """ Take a dye array and figure out where lanes are located.

        :Args:
            :param arr: A dye array (dyeEnd)

        :returns: A list of dictionaries with ('start', 'end')
        :rtype: list of dict

        """
        # Group consecutive pixels (aka lanes)
        lanes = self.consecutive(arr)

        # Iterate over pixel groups and pull out the boundries of the lane and the
        # midpoint
        coords = list()
        for lane in lanes:
            coords.append({'start': lane[0], 'end': lane[-1]})

        return coords

    def consecutive(self, arr, stepsize=1):
        """ Identify the location of consecutively numbered parts of an array.

        Found this solution at:
        http://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy 

        :Args:
            :param arr: 1D array
            :type arr: numpy.array
            :param stepsize: The distance you want to require between groups
            :type stepsize: int

        :returns: a list of numpy.arrays grouping consecutive number together
        :rtype: list of numpy.arrays

        """
        return np.split(arr, np.where(np.diff(arr) != stepsize)[0]+1)

    def getLadderLocation(self, sample):
        """ Identify lane(s) with ladder. 
        
        Using information from the sample sheet (description column), figure
        out which lane(s) contain the ladder. 

        """
        ladder = sample[sample['description'].str.lower() == 'ladder'].index
        if not ladder.any():
            logger.warn('There was no ladder present in the sample sheet. If a ' +
                        'ladder was used, please add "ladder" to the description field ' +
                        'in the sample sheet.')
            ladder = None

        return ladder

    def generateGrayIntensities(self):
        """ Get column vector of intensity values.

        From each lane pull the mean intensity.

        :Attr:
            :param self.laneIntensities: A list of grary scale intensities
            :type self.laneIntensities: list of float

        """
        intensities = list()
        for lane in self.laneLocations:
            start = lane[0]
            end = lane[1]
            midpoint = lane[3]
            col = self.grayGel[:, start:end]
            mean = np.mean(col, axis=1)
            intensities.append(mean)

        self.laneIntensities = intensities


class GelLane():
    """ Class representing a single lane of a gel """
    
    def __init__(self, gel, start, end, wellID, description=None):
        """ """
        # Set basic attributes
        self.start = start
        self.end = end
        self.wellID = wellID.upper()
        self.description = str(description).lower()

        # Calculate mean for the lane
        self.lane = gel[:, start:end]
        self.laneMean = self.lane.mean(axis=1)

        # If the lane is a ladder then set additional attributes
        if self.description == 'ladder':
            self.ladder = True
            self.getMW()

        else:
            self.ladder = False

    def getMW(self):
        """ Get molecular weights from a ladder. 
        
        Using a ladder lane, estimate the location of molecular weights.
        
        """
        # Molecular Weights defined by ladder
        weights = [1500, 1000, 700, 500, 400, 300, 200, 100, 50, 25]
        self.MW = None

        # Copy mean lane intensity
        his = self.laneMean.copy()

        # For the ladder there should be 10 peaks, iterate over a range of
        # filters until calling exactly 10 peaks.
        for i in np.arange(0, 1, 0.01):
            his[his < i] = 0
            peaks = signal.find_peaks_cwt(his, widths=np.arange(3, 5))
            if len(peaks) == 10:
                self.MW = zip(peaks, weights)
                break

        pickle.dump(self.MW, open('/Users/fearjm/devel/tapeAnalyst/data/mw.pkl', 'wb'))

        if self.MW is None:
            logger.warn('Peaks at all molecular weights could not be identified, check ladder.')



if __name__ == '__main__':
    pass
