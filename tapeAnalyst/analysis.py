#!/usr/bin/env python

# Built-in
import logging
logger = logging.getLogger()
import base64
from io import BytesIO

# 3rd Party
import numpy as np
from numpy.random import choice
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.signal as signal
import scipy.stats as stats
from skimage import exposure
from diptest.diptest import diptest as diptest


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


def callPeaks(lane, gain=7, hamming=5, filt=0.2, order=9):
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

        :param filt: Remove all pixels whose intensity is below this value.
        :type filt: float

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
        logger.warn('No Dye Front - Lane {}: {}'.format(lane.index, lane.wellID))

    try:
        dyeEndPeak = int(np.ceil(np.mean([lane.dyeEndStart, lane.dyeEndEnd])))
        laneDist[dyeEndPeak] = laneDist[dyeEndPeak] + 2
        laneDist[:dyeEndPeak-1] = 0
    except:
        logger.warn('No Dye End - Lane {}: {}'.format(lane.index, lane.wellID))

    # Filter out low levels
    laneDist[laneDist < filt] = 0

    # Find local maxima
    peaks = signal.argrelmax(laneDist, order=order)[0]

    return peaks


def callValleys(lane, gain=7, hamming=5, filt=0.2, order=9):
    """ Identify vallies in the lane 
    
    :Args:
        :param lane: The lane which to call valleys.
        :type lane: tapeAnalyst.gel_processing.GelLane

        :param gain: The gain value to use for increasing contrast (see
            skimage.exposure.adjust_sigmoid)
        :type gain: int

        :param hamming: The value to use Hamming convolution (see
            scipy.signal.hamming)
        :type gain: int

        :param filt: Remove all pixels whose intensity is below this value.
        :type filt: float

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
        logger.warn('No Dye Front - Lane {}: {}'.format(lane.index, lane.wellID))

    try:
        dyeEndPeak = int(np.ceil(np.mean([lane.dyeEndStart, lane.dyeEndEnd])))
        laneDist[dyeEndPeak] = laneDist[dyeEndPeak] + 2
        laneDist[:dyeEndPeak-1] = 0
    except:
        logger.warn('No Dye End - Lane {}: {}'.format(lane.index, lane.wellID))

    # Filter out low levels
    laneDist[laneDist < filt] = 0

    # Find local maxima
    valleys = signal.argrelmin(laneDist)[0]

    return valleys


def dip(peak):
    """ Run a dip test. 

    The diptest can be used to test if a distribution is unimodal. In order to
    get it to work, I have to turn the peak signal into a distribution by
    simulating, and then run the test on the simulated data. This is a little
    hackish, there is probably a better/faster way.
    
    """
    # Normalize the peak section to sum to 1
    norm = peak / peak.sum()

    # Simulate data from the peak distribution
    sim = choice(np.arange(peak.shape[0]), p=norm, size=10000)

    # Run diptest
    test, pval = diptest(sim)

    return test, pval


def summaryStats(gel, lane, peaks, valleys, molRange):
    """ Calculate various summary stats based on distribution of peaks. 
    
    A good library should have 3 peaks. The two dyes and a middle peak that
    is a smear in the desired range. Check for the number of peaks, and if
    lane appears to be good analyze the distribution of the middle peak and
    determine if approximately normal.
    
    """
    # Covert coordinates to molecular weights
    if gel.ladders is not None:
        # Iterpolate from coordinates to MW
        iToMW = interpToMW(gel.ladders)

        # Convert peaks to molecular weights
        peaksMW = np.sort(iToMW(peaks))

        # Convert valleys to molecular weights
        valleysMW = np.sort(iToMW(valleys))
    else:
        peaksMW = None
        valleysMW = None

    if len(peaksMW) == 3:
        # If there are only 3 peaks, than distribution is unimodal.
        lane.flag.append('UNIMODAL')
        peakLoc = peaksMW[1]

        if molRange is not None:
            if (peaksMW[1] < molRange).any() and (peaksMW[1] > molRange).any():
                lane.flag.append('PEAKINRANGE')

        try:
            # Try to find the valley surrounding main peak
            leftValley = int(valleys[valleysMW < peaksMW[1]][-1])
            rightValley = int(valleys[valleysMW > peaksMW[1]][0])

            # Slice peak for diptest
            section = lane.laneMean[leftValley:rightValley]
        except:
            section = lane.laneMeanNoDye

        # Test for normality, I don't think the nature of the gel peaks
        # lends itself to normality assumptions. In general gels are left
        # skewed.
        normTest, normPval = stats.shapiro(section)
        if normPval > 0.05:
            lane.flag.append('NORMAL')
        else:
            lane.flag.append('NONORM')
    else:
        section = None
        lane.flag.append('MULTIMODAL')
        normTest = np.nan
        normPval = np.nan
        peakLoc = ';'.join([str(x) for x in peaksMW[1:-1]])


    # Only run stats is both dye fronts are present
    if not 'NODYEEND' in lane.flag and not 'NODYEFRONT' in lane.flag:
        # Run diptest for unimodality
        if section is not None:
            dipTval, dipPval = dip(section)
        else:
            dipTval, dipPval = dip(lane.laneMeanNoDye)

        # Basic stats
        std = np.std(lane.laneMeanNoDye)
        skew = stats.skew(lane.laneMeanNoDye)
        dipTest = '{} ({})'.format(np.round(dipTval, 3), dipPval)
        shapiro = '{} ({})'.format(np.round(normTest, 3), normPval)

        # Build table of stats
        summaryStatsTable = pd.DataFrame([peakLoc, std, skew, dipTest, shapiro],
                index=['Peak(s) Molecular Weight', 'Std Dev', 'Skewness', 'Diptest for Unimodality', 'Normality Test (Shapiro-Wilk)'], columns=['Values'])

    else:
        # Build empty table missing dyes
        summaryStatsTable = pd.DataFrame(index=['Peak(s) Molecular Weight', 'Std Dev', 'Skewness', 'Diptest for Unimodality', 'Normality Test (Shapiro-Wilk)'], columns=['Values'])

    return summaryStatsTable


def addMW_Y(ax, MW):
    ax.yaxis.tick_right()
    ax.set_yticks(np.array(MW)[:,0])
    ax.set_yticklabels(np.array(MW)[:,1])


def addMW_X(ax, MW):
    ax.set_xticks(np.array(MW)[:,0])
    ax.set_xticklabels(np.array(MW)[:,1], rotation=90)


def convertImage(fig):
    """ Converts an image into its base64 byte representation """
    figfile = BytesIO()
    fig.savefig(figfile, format='png')

    return base64.b64encode(figfile.getvalue()).decode('utf8')


def summarizeDistributions(args, gel):
    """ Summarize distributions for each lane.  """
    # Generate a spacer for plotting later.
    spacer = np.zeros((gel.grayGel.shape[0], 10))

    if gel.ladders is not None:
        # Build a 'gel' image will all of the ladder lanes put together.
        ladderImg = list()
        ladderImg.append(spacer)
        for ladder in gel.ladders:
            ladderImg.append(ladder.lane)
            ladderImg.append(spacer)
        ladderImg.append(spacer)
        ladderImg = np.column_stack(ladderImg)
        
        # Set MW
        MW = np.array(gel.ladders[0].MW)
    else:
        MW = None

    # Iterate over each lane and summarize
    results = list()
    for lane in gel.lanes:
        # Call peaks for each lane
        peaks = callPeaks(lane, gain=args.gain, hamming=args.hamming, filt=args.filter, order=args.order)

        # Call valleys for each lane
        valleys = callValleys(lane, gain=args.gain, hamming=args.hamming, filt=args.filter, order=args.order)

        # Basic summary stats
        summaryTable = summaryStats(gel, lane, peaks, valleys, args.range)

        # Generate Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        # Gel Image
        img = np.column_stack([spacer, lane.lane, spacer, spacer, ladderImg])
        ax1.imshow(img, cmap='Greys')
        ax1.get_xaxis().set_visible(False)

        # Density Plot
        ax2.plot(lane.laneMean, color='k')
        for peak in peaks[1:-1]:
            ax2.axvline(peak, color='r', ls=':', lw=2)

        # If a range was given, highlight
        if args.range is not None:
            # build interplater from MW to coordinates
            iFromMW = interpFromMW(gel.ladders)
            molRange = iFromMW(args.range)

            molStart = int(min(molRange))
            molEnd = int(max(molRange))
            ax2.fill_between(range(molStart, molEnd), 0, lane.laneMean[molStart:molEnd], color='g', alpha=0.3)

        # Add MW is Ladder is present
        if MW is not None:
            addMW_Y(ax1, MW)
            addMW_X(ax2, MW)
        
        # Convert fig to binary
        fig64 = convertImage(fig)
        plt.close()

        # Build result output
        results.append({'image': fig64, 'flags': lane.flag, 
                        'wellID': lane.wellID, 'description': lane.description,
                        'table': summaryTable.to_html(), 'index': lane.index})

    return results
            

def fullGelImage(gel):
    """ Create the full color image of gel.

    Plot the full color image of the gel and then convert to binary for
    embeding in HTML.
    
    """
    gelFig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(gel.colorGel)
    ax.get_xaxis().set_visible(False)
    if gel.ladders is not None:
        addMW_Y(ax, gel.ladders[0].MW)
    gelFig64 = convertImage(gelFig)
    plt.close()

    return gelFig64


if __name__ == '__main__':
    pass
