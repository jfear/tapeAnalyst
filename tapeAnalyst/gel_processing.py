#!/usr/bin/env python

import logging
logger = logging.getLogger()

import numpy as np
import pandas as pd
from scipy import signal as signal
import scipy.stats as stats
from scipy.interpolate import interp1d
from skimage import io, color, exposure


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

        :param self.dyeGel: Gel image containing only the dye front and dye
            end.
        :type self.dyeGel: numpy.ndarray

        :param self.gelTop: Location of the top of the gel (Purple band).
        :type self.gelTop: int

        :param self.gelBottom: Location of the bottom of the gel (Green band).
        :type self.gelBottom: int

        :param self.gelLeft: Location of the left of the gel (Green and Purple band).
        :type self.gelLeft: int

        :param self.gelRigth: Location of the right of the gel (Green and Purple band).
        :type self.gelRigth: int

        :param self.lanes: A list containing each separate lane from the gel
            (excluding ladders).
        :type self.lanes: list of tapeAnalyst.gel_processing.GelLane

        :param self.lanes: A list containing all lanes that were annotated as
            ladders.
        :type self.lanes: list of tapeAnalyst.gel_processing.GelLane

    """
    def __init__(self, fname, sample):
        # read gel image
        self.colorGel = io.imread(fname)

        # Convert gel to gray scale for intensity analysis
        self.grayGel = 1 - color.rgb2gray(self.colorGel)

        # Figure out the top and bottom coordinates of gel using the dye
        self.getGelRegion()

        # Parse apart lanes using dye front and end and pull out lanes
        # containing ladders.
        self.splitLanes(sample)

        if len(self.ladders) == 0:
            self.ladders = None

    def getGelRegion(self):
        """ Identify the region in the image that contains the gel.

        Using the dye front (Green) and the dye end (Purple) figure out the
        coordinates in the image that contain the gel. Select the regions with
        "green" or "purple" colors. 

        """
        # Get bottom coords from dye front (GREEN).
        self.dyeFrontGel = self.dyeMarker(dye='front')
        dyeFrontCoords = np.nonzero(self.dyeFrontGel.sum(axis=1))[0]
        if dyeFrontCoords[-1].any():
            self.gelBottom = dyeFrontCoords[-1]
        else:
            self.gelBottom = None

        # Get top coords from dye end (PURPLE)
        self.dyeEndGel = self.dyeMarker(dye='end')
        dyeEndCoords = np.nonzero(self.dyeEndGel.sum(axis=1))[0]
        if dyeEndCoords[0].any():
            self.gelTop = dyeEndCoords[0]
        else:
            self.gelTop = None

        # Get left and right coords by combining green and purple
        self.dyeGel = self.dyeFrontGel + self.dyeEndGel
        self.dyeGel[self.dyeGel > 0] = 1
        self.grayGel[self.dyeGel > 0] = 1

        dyeCoords = np.nonzero(self.dyeGel.sum(axis=0))[0]
        self.gelLeft = dyeCoords[0]
        self.gelRight = dyeCoords[-1]

    def dyeMarker(self, dye='front'):
        """ Set boolean mask on gel image to identify dye front or dye end.

        The TapeStation annotates the dye front as green and the dye end as
        purple.  Using these color profiles, mask the remainder of the image to
        be white, effectively selecting the regions with these colors.

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
        """ Using the dye front and end, separate lanes. """
        # Create row vector where lanes are values above 0
        dyeLanes = np.nonzero(self.dyeGel.sum(axis=0))[0]

        # Group consecutive pixels that are separated by 0's into lanes
        laneLocations = self.getLaneLocations(dyeLanes)

        # Iterate of lanes and create Lane objects
        self.laneIndex = dict()
        self.lanes = list()
        self.ladders = list()
        for index, lane in enumerate(laneLocations):
            row = sample.iloc[index]
            glane = GelLane(self, index=index, wellID=row['wellID'], description=row['description'], **lane)
            if 'LADDER' in glane.flag:
                self.ladders.append(glane)
            else:
                self.laneIndex[row['wellID']] = index
                self.lanes.append(glane)

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

    def getLane(self, wellID):
        """ Given a wellID number, pull specific lane. 
        
        :param wellID: ID number of the well that you want to view
        :type wellID: str

        """
        return self.lanes[self.laneIndex[wellID]]


class GelLane():
    """ A single lane of a tape station gel gel. 
    
    :Args:
        :param gel: Processed tape station gel image.
        :type gel: tapeAnalyst.gel_processing.TapeStationGel

        :param start: Start location of the lane
        :type start: int

        :param end: End location of the lane
        :type end: int

        :param wellID: Sample well ID from a 96 well plate
        :type wellID: str

        :param description: Sample description, keyword ladder if the sample is
            a ladder.
        :type description: str
    
    :Attr:
        :param self.start: Start location of the lane
        :type self.start: int

        :param self.end: End location of the lane
        :type self.end: int

        :param self.wellID: Sample well ID from a 96 well plate
        :type self.wellID: str

        :param self.description: Sample description, keyword ladder if the sample is
            a ladder.
        :type self.description: str

        :param self.lane: 2d array of gray scale intensity values.
        :type self.lane: numpy.ndarray

        :param self.laneMean: Column vector of mean intensity values for the lane.
        :type self.lane: list of float

        :param self.flag: A set of flags describing the quality of the lane.
            * LADDER: if the lane is a ladder
            * MULTIMODAL: if the lane has multiple peaks, suggesting poor quality
            * NODYEEND: if the dye end was not present
            * NODYEFRONT: if the dye front was not present
            * NONORM: if the lane has a single large peak but it is not normally distributed.
            * NORMAL: if the lane has a single large peak that is normally distributed.
            * UNIMODAL: if the lane has single peaks
        :type self.flag: list of str

        :param self.MW: If the lane is a ladder, then identify intensity peaks
            from self.laneMean and relate these to the ladders molecular
            weights by creating a list of tuples (peak location, molecular
            weight).
        :type self.MW: list of tuples of int

        :param self.peaks: List of peak locations.
        :type self.peaks: list of int

        :param self.valleys: List of vally locations.
        :type self.valleys: list of int

    """
    def __init__(self, gel, start, end, wellID, index, description=None):
        # Set basic attributes
        self.start = start
        self.end = end
        self.wellID = wellID.upper()
        self.index = index + 1
        self.description = str(description).lower()
        self.flag = list()

        # Identify the lane level dye locations
        self.getDyeLocations(gel)

        # Calculate mean pixel intensity for each row in the lane
        self.lane = gel.grayGel[:, start:end]
        self.laneMean = self.lane.mean(axis=1)

        # Make a filtered distribution excluding the Dyes
        if self.dyeEndEnd and self.dyeFrontStart:
            self.laneMeanNoDye = self.laneMean[self.dyeEndEnd+15:self.dyeFrontStart-15]

        # If the lane is a ladder then set additional attributes
        if self.description == 'ladder':
            self.flag.append('LADDER')
            self.getMW()

    def getDyeLocations(self, gel):
        """ Locate where the dye end and dye front are at in the lane. """
        # Dye Front (Green)
        front = gel.dyeFrontGel[:, self.start:self.end]
        locsFront = np.nonzero(front.sum(axis=1))[0]
        try:
            self.dyeFrontStart = locsFront[0]
            self.dyeFrontEnd = locsFront[-1]
        except:
            self.dyeFrontStart = None
            self.dyeFrontEnd = None
            self.flag.append('NODYEFRONT')

        # Dye End (Purple)
        end = gel.dyeEndGel[:, self.start:self.end]
        locsEnd = np.nonzero(end.sum(axis=1))[0]
        try:
            self.dyeEndStart = locsEnd[0]
            self.dyeEndEnd = locsEnd[-1]
        except:
            self.dyeEndStart = None
            self.dyeEndEnd = None
            self.flag.append('NODYEEND')

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
            self.peaks = signal.find_peaks_cwt(his, widths=np.arange(3, 5))
            if len(self.peaks) == 10:
                self.MW = list(zip(self.peaks, weights))
                break

        if self.MW is None:
            logger.warn('Peaks at all molecular weights could not be identified, check ladder.')


def interpToMW(ladders):
    """ Create a MW interpolation method if ladder is present. 

    Convert from coordinate system to molecular weight.

    :Args:
        :param ladders: List of lanes that are ladders.
        :type ladders: list of tapeAnalyst.gel_processing.GelLane

    :returns: An interpolation function.
    :rtype: scipy.interpolate.interp1d

    """
    # Get vector of gel locations by averaging MW locations across
    # ladders
    x = np.array(ladders[0].MW)[:, 0]
    y = np.array(ladders[0].MW)[:, 1]

    # Build interpolation function to estimate MW for any position
    # value. MWs are logarithmic, use a quadratic function.
    return interp1d(x, y, kind='quadratic', bounds_error=False)

def interpFromMW(ladders):
    """ Create a MW interpolation method if ladder is present. 

    Convert from molecular weight to coordinate system.

    :Args:
        :param ladders: List of lanes that are ladders.
        :type ladders: list of tapeAnalyst.gel_processing.GelLane

    :returns: An interpolation function.
    :rtype: scipy.interpolate.interp1d

    """
    # Get vector of gel locations by averaging MW locations across
    # ladders
    x = np.array(ladders[0].MW)[:, 1]
    y = np.array(ladders[0].MW)[:, 0]

    # Build interpolation function to estimate coordinate from MW.
    return interp1d(x, y, kind='quadratic', bounds_error=False)


def getFirst(array, value, direction='greater'):
    """ Get first value that is either greater than or less than a value.

    Given an array and a value. Figure out where in the array the first value
    that is greater than or the last value that is less than. Can be used to
    find bands in an image.

    :Args:
        :param array: List of numbers.
        :type array: array-like

        :param value: value you want to test.
        :param value: int

        :param direction: Direction for the test 'greater' or 'less'.
        :type direction: str

    :returns: Index of the first value.
    :rtype: int

    """
    if direction == 'greater':
        mask = array > value
        return array[np.nonzero(mask)[0][0]]
    elif direction == 'less':
        mask = array < value
        return array[np.nonzero(mask)[0][-1]]


def callPeaks(lane, gain=7, hamming=5, filter=0.2, order=9):
    """ Identify peaks in the lane 
    
    :Args:
        :param lane: The lane which to call peaks.
        :type lane: tapeAnalyst.gel_processing.GelLane

        :param gain: The gain value to use for increasing contrast (see
            skimage.exposure.adjust_sigmoid)
        :type gain: int

        :param hamming: The value to use Hamming convolution (see
            scipy.signal.hamming)
        :type gain: int

        :param filter: Remove all pixels whose intensity is below this value.
        :type filter: float

        :param order: The distance allowed for finding maxima (see scipy.signal.argrelmax)
        :type order: int

    """
    # Increase contrast to help with peak calling
    ladj = exposure.adjust_sigmoid(lane.lane, cutoff=0.5, gain=gain)

    # Tack the max pixel intensity for each row in then lane's gel image.
    laneDist = ladj.max(axis=1)

    # Smooth the distribution
    laneDist = signal.convolve(laneDist, signal.hamming(hamming))

    # Get the locations of the dye front and dye end. Peak calling is difficult
    # here because dyes tend to plateau. To aid peak calling, add an artificial
    # spike in dye regions. Also remove all peaks outside of the dyes
    try:
        dyeFrontPeak = int(np.ceil(np.mean([lane.dyeFrontStart, lane.dyeFrontEnd])))
        laneDist[dyeFrontPeak] = laneDist[dyeFrontPeak] + 2
        laneDist[dyeFrontPeak+1:] = 0
    except:
        logger.warn('No Dye Front')

    try:
        dyeEndPeak = int(np.ceil(np.mean([lane.dyeEndStart, lane.dyeEndEnd])))
        laneDist[dyeEndPeak] = laneDist[dyeEndPeak] + 2
        laneDist[:dyeEndPeak-1] = 0
    except:
        logger.warn('No Dye End')

    # Filter out low levels
    laneDist[laneDist < filter] = 0

    # Find local maxima
    peaks = signal.argrelmax(laneDist, order=order)[0]

    return peaks


def summaryStats(lane, peaks, molRange):
    """ Calculate various summary stats based on distribution of peaks. 
    
    A good library should have 3 peaks. The two dyes and a middle peak that
    is a smear in the desired range. Check for the number of peaks, and if
    lane appears to be good analyze the distribution of the middle peak and
    determine if approximately normal.
    
    """
    if len(peaks) == 3:
        # If there are only 3 peaks, than distribution is unimodal.
        lane.flag.append('UNIMODAL')

        # See if the peak is in my molecular range
        if molRange is not None:
            if (peaks[1] < molRange).any() and (peaks[1] > molRange).any():
                lane.flag.append('PEAKINRANGE')

        # Test for normality, I don't think the nature of the gel peaks
        # lends itself to normality assumptions. In general gels are left
        # skewed.
        test, pval = stats.shapiro(lane.laneMeanNoDye)
        if pval > 0.05:
            lane.flag.append('NORMAL')
        else:
            lane.flag.append('NONORM')
    else:
        lane.flag.append('MULTIMODAL')

    # Summary stats for density excluding dyes
    if not 'NODYEEND' in lane.flag and not 'NODYEFRONT' in lane.flag:
        mean = np.mean(lane.laneMeanNoDye)
        median = np.median(lane.laneMeanNoDye)
        mode = stats.mode(lane.laneMeanNoDye)[0]
        std = np.std(lane.laneMeanNoDye)
        skew = stats.skew(lane.laneMeanNoDye)

        # Build table of stats
        summaryStatsTable = pd.DataFrame([mean, median, mode, std, skew], index=['Mean', 'Median', 'Mode', 'Std Dev', 'Skewness'], columns=['Values'])
        summaryStatsTable.index.name = 'Measure'

        # Test if main part of the distribution is in MW region
        if molRange is not None:
            if (mean < molRange).any() and (mean > molRange).any():
                lane.flag.append('MEANINRANGE')
            if (median < molRange).any() and (median > molRange).any():
                lane.flag.append('MEDIANINRANGE')
    else:
        # Build empty table missing dyes
        summaryStatsTable = pd.DataFrame(index=['Mean', 'Median', 'Mode', 'Std Dev', 'Skewness'], columns=['Values'])
        summaryStatsTable.index.name = 'Measure'


    return summaryStatsTable






if __name__ == '__main__':
    pass
