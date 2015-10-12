#!/usr/bin/env python
import numpy as np
from skimage import io, color

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
    def __init__(self, fname, ladder=None, cropPadding=20):
        # read gel image
        self.colorGel = io.imread(fname)

        # Convert gel to gray scale for intensity analysis
        self.grayGel = 1 - color.rgb2gray(self.colorGel)

        # Save the lane number with the ladder when given
        self.ladder = ladder

        # Figure out the top and bottom coordinates of gel using the dye
        ## Get coords for dye front (GREEN)
        dyeFrontGel = self.dyeMarker(dye='front')
        dyeFrontCoords = np.nonzero(dyeFrontGel[:, :].sum(axis=1))[0]

        if dyeFrontCoords[-1].any():
            self.gelBottom = dyeFrontCoords[-1]
        else:
            self.gelBottom = None

        ## Get coords for dye end (PURPLE)
        dyeEndGel = self.dyeMarker(dye='end')
        dyeEndCoords = np.nonzero(dyeEndGel[:, :].sum(axis=1))[0]
        if dyeEndCoords[0].any():
            self.gelTop = dyeEndCoords[0]
        else:
            self.gelTop = None

        ## Get left and right coords by comibining green and purple
        dyeGel = dyeFrontGel + dyeEndGel
        dyeCoords = np.nonzero(dyeGel[:, :].sum(axis=0))[0]
        self.gelLeft = dyeCoords[0]
        self.gelRight = dyeCoords[-1]

        # Locate lanes using dye end
        dyeLanes = np.nonzero(dyeGel.sum(axis=0))[0]
        self.laneLocations = self.getLaneLocations(dyeLanes)

    def dyeMarker(self, dye='front'):
        """ Set boolean mask on gel image to easily identify dye front or dye end. 

        The TapeStation annotates the dye front as green and the dye end as purple.
        Using these color profiles, mask the remainder of the image to be white.

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

    def getLaneLocations(self, arr):
        """ Take a dye array and figure out where lanes are located.

        :Args:
            :param arr: A dye array (dyeEnd)

        :returns: A list of tuples with (start, end, lane width, midpoint of the lane)
        :rtype: list of tuples

        """
        # Group consecutive pixels (aka lanes)
        lanes = self.consecutive(arr)

        # Iterate over pixel groups and pull out the boundries of the lane and the
        # midpoint
        coords = list()
        for lane in lanes:
            start = lane[0]
            end = lane[-1]
            width = end - start
            midpoint = int(np.ceil(width / 2 + start))
            coords.append((start, end, width, midpoint))

        return coords

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

if __name__ == '__main__':
    pass
