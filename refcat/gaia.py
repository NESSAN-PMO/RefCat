#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import astropy
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import ascii
import os, sys
import glob
from astropy import log
import healpy as hp
import os.path

class GAIA1:

    def __init__( self, path = None, fmt = 'fits', lvl = 6 ):
        if path == None:
            self.path = os.environ.get( "GAIA1_PATH" )
        else:
            self.path = path
        if self.path == None:
            log.error( "GAIA1 path is not set." )
            raise IOError( "None type path." )
        if len( glob.glob( os.path.join( self.path, fmt, "GaiaSource*" ) ) ) == 0:
            log.error( "No GAIA1 data file found." )
            self.valid = 0
            raise IOError( "Invalid path." )
        else:
            self.valid = 1
        self.fmt = fmt
        self.LEVEL = lvl
        self.NSIDE = 2 ** self.LEVEL
        selfpath = os.path.dirname( __file__ )
        self.index = Table.read( os.path.join( selfpath, "data", "gaia1index" ), format = 'ascii' )
        self.data = Table()


    def _get_gaia1_zone_file( self, id_min, id_max ):

        files = []
        for i in range( len( id_min ) ):
            for row in self.index:
                if id_min[i] > row['col3']:
                    continue
                elif id_max[i] < row['col2']:
                    break
                if self.fmt == 'csv':
                    files.append( os.path.join( self.path, self.fmt, row['col1'] + '.csv.gz' ) )
                if self.fmt == 'fits':
                    files.append( os.path.join( self.path, self.fmt, row['col1'] + '.fits' ) )
                if self.fmt == 'votable':
                    files.append( os.path.join( self.path, self.fmt, row['col1'] + '.vot.gz' ) )
        files = Table( [files] )
        files = astropy.table.unique( files )
        return( files )

    def _output_catalog( self, stars, ids, keep = 0 ):
        pass

    def extract( self, ra, dec, width, height, keep = 0 ):
        if not self.valid:
            return
        if keep == 0:
            self.data = Table()

        polygon = SkyCoord( [ra-width/2., ra+width/2., ra+width/2., ra-width/2.],\
                [dec-height/2., dec-height/2., dec+height/2., dec+height/2.],\
                unit = 'deg' ).cartesian.get_xyz().T
        pix = hp.query_polygon( self.NSIDE, polygon, inclusive = True, nest = True )
        source_id_min = pix * 2 ** 35 * 4 ** ( 12 - self.LEVEL )
        source_id_max = ( pix + 1 ) * 2 ** 35 * 4 ** ( 12 - self.LEVEL )
        filelist = self._get_gaia1_zone_file( source_id_min, source_id_max )

        for f in filelist:
            catalog = Table.read( f['col0'], format = self.fmt )
            p = catalog[ np.where( ( catalog['ra'] > ra - width/2. )\
                    & ( catalog['ra'] < ra + width/2. )\
                    & ( catalog['dec'] > dec - height/2. )\
                    & ( catalog['dec'] < dec + height/2. ) ) ]
            self.data = astropy.table.vstack( [self.data, p] )
        return( len( self.data ) )




class TGASPTYC:

    def __init__( self, path = None, datafile = 'tgasptyc.dat', lvl = 4 ):
        if path == None:
            self.path = os.environ.get( "TGASPTYC_PATH" )
        else:
            self.path = path
        if self.path == None:
            log.error( "TGASPTYC path is not set." )
            raise IOError( "None type path." )
        self.datafile = os.path.join( self.path, datafile )
        self.readmefile = os.path.join( self.path, "ReadMe" )
        self.linelength = len( open( self.datafile ).readline() )

        if not os.path.isfile( self.datafile ):
            log.error( "No TGASPTYC data file found." )
            self.valid = 0
            raise IOError( "Invalid path." )
        else:
            self.valid = 1
        self.LEVEL = lvl
        self.NSIDE = 2 ** self.LEVEL
        selfpath = os.path.dirname( __file__ )
        self.index = np.loadtxt( os.path.join( selfpath, "data", "tgasindex_l{}".format( lvl ) ), dtype = np.int64 )
        self.data = Table()


    def _get_tgasptyc_zone_file( self, pixels):

        ranges = []
        for p in pixels:
            ranges.append( ( self.index[p], self.index[p+1] ) )
        return( ranges )

    def _output_catalog( self, stars, ids, keep = 0 ):
        pass

    def extract( self, ra, dec, width, height, keep = 0 ):
        if not self.valid:
            return
        if keep == 0:
            self.data = Table()

        polygon = SkyCoord( [ra-width/2., ra+width/2., ra+width/2., ra-width/2.],\
                [dec-height/2., dec-height/2., dec+height/2., dec+height/2.],\
                unit = 'deg' ).cartesian.get_xyz().T
        pix = hp.query_polygon( self.NSIDE, polygon, inclusive = True, nest = True )
        rangelist = self._get_tgasptyc_zone_file( pix )

        f = open( self.datafile, "r" )
        lines = []
        for r in rangelist:
            f.seek( r[0] * self.linelength )
            lines.append( f.read( ( r[1] - r[0] ) * self.linelength ) )
        f.close()
        content = ''.join( lines )

        reader = ascii.get_reader( Reader = ascii.Cds, fill_values = [('',0)], readme = self.readmefile )
        reader.data.table_name = "tgasptyc.dat"
        catalog = reader.read( content )
        p = catalog[ np.where( ( catalog['RAdeg'] > ra - width/2. )\
                & ( catalog['RAdeg'] < ra + width/2. )\
                & ( catalog['DEdeg'] > dec - height/2. )\
                & ( catalog['DEdeg'] < dec + height/2. ) ) ]
        self.data = astropy.table.vstack( [self.data, p] )

        return( len( self.data ) )


if __name__ == "__main__":

    import time
    t = time.time()
    #cat = GAIA1()
    cat = TGASPTYC()
    print( cat.extract( 340.919, 30.922, 1, 1 ) )
    print( time.time() - t )
    print( cat.data )
