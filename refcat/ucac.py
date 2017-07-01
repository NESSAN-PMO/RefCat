#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import astropy
from astropy.table import Table, hstack, vstack
import os, sys
import glob
from astropy import log

class UCAC4:

    def __init__( self, path = None ):
        if path == None:
            self.path = os.environ.get( "UCAC4_PATH" )
        else:
            self.path = path
        if self.path == None:
            log.error( "UCAC4 path is not set." )
            raise IOError( "None type path." )
        if len( glob.glob( os.path.join( self.path, "z???" ) ) ) == 0:
            log.error( "No UCAC4 binary file found." )
            self.valid = 0
            raise IOError( "Invalid path." )
        else:
            self.valid = 1
        self.indexfile = os.path.join( self.path, "..", "u4i", "u4index.asc" )
        if not os.path.isfile( self.indexfile ):
            log.warning( "No UCAC4 index file found." )
            self.indexfile = None
        self.data = Table()
        self.UCAC4_RAW = np.dtype( [ 
            ('ra', np.int32), ('spd', np.int32),
            ('mag1', np.uint16), ('mag2', np.uint16),
            ('mag_e', np.uint8),
            ('obj_type', np.uint8), ('double_star_flag', np.uint8),
            ('ra_e', np.int8), ('dec_e', np.int8 ),
            ('n_ucac_total', np.uint8),
            ('n_ucac_used', np.uint8),
            ('n_cats_used', np.uint8),
            ('epoch_ra', np.uint16),
            ('epoch_dec', np.uint16),
            ('pm_ra', np.int16),
            ('pm_dec', np.int16),
            ('pm_ra_e', np.int8),
            ('pm_dec_e', np.int8),
            ('twomass_id', np.uint32),
            ('mag_j',np.uint16), ('mag_h',np.uint16), ('mag_k',np.uint16), 
            ('icq_flag_j', np.uint8),
            ('icq_flag_h', np.uint8),
            ('icq_flag_k', np.uint8),
            ('e2mpho_j', np.uint8),
            ('e2mpho_h', np.uint8),
            ('e2mpho_k', np.uint8),
            ('apass_mag_b', np.uint16),
            ('apass_mag_v', np.uint16),
            ('apass_mag_g', np.uint16),
            ('apass_mag_r', np.uint16),
            ('apass_mag_i', np.uint16),
            ('apass_mag_e_b', np.int8),
            ('apass_mag_e_v', np.int8),
            ('apass_mag_e_g', np.int8),
            ('apass_mag_e_r', np.int8),
            ('apass_mag_e_i', np.int8),
            ('yale_gc_flags', np.uint8),
            ('catalog_flags', np.uint32),
            ('leda_flag', np.uint8),
            ('twomass_ext_flag', np.uint8),
            ('id_number', np.uint32),
            ('ucac2_zone', np.uint16),
            ('ucac2_number', np.uint32)
            ] )
        self.UCAC4_STAR = np.dtype( [ 
            ('ra', np.float), ('spd', np.float),
            ('mag1', np.float), ('mag2', np.float),
            ('mag_e', np.float),
            ('obj_type', np.uint8), ('double_star_flag', np.uint8),
            ('ra_e', np.int), ('dec_e', np.int ),
            ('n_ucac_total', np.uint8),
            ('n_ucac_used', np.uint8),
            ('n_cats_used', np.uint8),
            # 12
            ('epoch_ra', np.float),
            ('epoch_dec', np.float),
            ('pm_ra', np.float),
            ('pm_dec', np.float),
            ('pm_ra_e', np.float),
            ('pm_dec_e', np.float),
            ('twomass_id', np.uint32),
            # 19
            ('mag_j',np.float), ('mag_h',np.float), ('mag_k',np.float), 
            ('icq_flag_j', np.uint8),
            ('icq_flag_h', np.uint8),
            ('icq_flag_k', np.uint8),
            ('e2mpho_j', np.float),
            ('e2mpho_h', np.float),
            ('e2mpho_k', np.float),
            # 28
            ('apass_mag_b', np.float),
            ('apass_mag_v', np.float),
            ('apass_mag_g', np.float),
            ('apass_mag_r', np.float),
            ('apass_mag_i', np.float),
            ('apass_mag_e_b', np.float),
            ('apass_mag_e_v', np.float),
            ('apass_mag_e_g', np.float),
            ('apass_mag_e_r', np.float),
            ('apass_mag_e_i', np.float),
            # 38
            ('yale_gc_flags', np.uint8),
            ('catalog_flags', np.uint32),
            ('leda_flag', np.uint8),
            ('twomass_ext_flag', np.uint8),
            ('id_number', np.uint32),
            ('ucac2_zone', np.uint16),
            ('ucac2_number', np.uint32)
            ] )


    def _get_ucac4_zone_file( self, zone_number, path ):
        subfolder = "u4{0}".format( 'n' if zone_number >= 380 else 's' )
        filename = "z{0:03d}".format( zone_number )
        if os.path.isfile( os.path.join( path, filename ) ):
            return os.path.join( path, filename )
        elif os.path.isfile( os.path.join( path, subfolder, filename ) ):
            return os.path.join( path, subfolder, filename )
        else:
            return None

    def _get_index_file_offset( self, zone, ra_start ):
        rval = ( zone - 1 ) * ( 1440 * 21 + 6 ) + ra_start * 21
        if ( ra_start ):
            rval += 6
        return rval

    def _output_catalog( self, stars, ids, keep = 0 ):
        if stars == []:
            return
        stars = np.array( stars ).astype( self.UCAC4_STAR )
        stars['ra'] /= 3600000.
        stars['spd'] -= 324000000.
        stars['spd'] /= 3600000.
        stars['ra_e'] += 128
        stars['dec_e'] += 128
        stars['mag1'] /= 1000.
        stars['mag2'] /= 1000.
        stars['mag_e'] /= 100.
        stars['epoch_ra'] /= 100.
        stars['epoch_ra'] += 1900.
        stars['epoch_dec'] /= 100.
        stars['epoch_dec'] += 1900.
        stars['pm_ra'] /= 10.
        stars['pm_dec'] /= 10.
        stars['pm_ra_e'] += 128.
        stars['pm_ra_e'] /= 10.
        stars['pm_dec_e'] += 128.
        stars['pm_dec_e'] /= 10.
        stars['mag_j'] /= 1000.
        stars['mag_h'] /= 1000.
        stars['mag_k'] /= 1000.
        stars['e2mpho_j'] /= 100.
        stars['e2mpho_h'] /= 100.
        stars['e2mpho_k'] /= 100.
        stars['apass_mag_b'] /= 1000.
        stars['apass_mag_v'] /= 1000.
        stars['apass_mag_g'] /= 1000.
        stars['apass_mag_r'] /= 1000.
        stars['apass_mag_i'] /= 1000.
        stars['apass_mag_e_b'] /= 100.
        stars['apass_mag_e_v'] /= 100.
        stars['apass_mag_e_g'] /= 100.
        stars['apass_mag_e_r'] /= 100.
        stars['apass_mag_e_i'] /= 100.
        stars = Table( stars )
        ids = Table( np.array( ids ), names = ( 'zone', 'offset' ) )
        stars = hstack( [stars, ids] )
        stars.rename_column( 'spd', 'dec' )

        if keep:
            if not isinstance( self.data, Table ):
                self.data = Table()
            self.data = vstack( [self.data, stars] )
        else:
            self.data = stars

    def extract( self, ra, dec, width, height, keep = 0 ):
        if not self.valid:
            return
        tmpstars = []
        tmpids = []
        rval = 0
        ra1 = ra - width / 2.
        ra2 = ra + width / 2.
        dec1 = dec - height / 2.
        dec2 = dec + height / 2.
        zone_height = .2
        zone = int( ( dec1 + 90. ) / zone_height ) + 1
        end_zone = int( ( dec2 + 90. ) / zone_height ) + 1
        index_ra_resolution = 1440
        ra_start = int( ra1 * index_ra_resolution / 360. )
        buffsize = 400
        if ( zone < 1 ):
            zone = 1
        if ( ra_start < 0 ):
            ra_start = 0

        while( rval >= 0 and zone <= end_zone ):
            try:
                ifile = open( self._get_ucac4_zone_file( zone, self.path ), 'rb' )
            except Exception as e:
                log.warning( "Zone file open failed," + e )
                ifile = None
            if ( ifile ):
                keep_going = 1
                max_ra = int( ra2 * 3600. * 1000. )
                min_ra = int( ra1 * 3600. * 1000. )
                min_spd = int( ( dec1 + 90. ) * 3600. * 1000. )
                max_spd = int( ( dec2 + 90. ) * 3600. * 1000. )
                acceptable_limit = 40
                index_file_offset = self._get_index_file_offset( zone, ra_start )
                cached_index_data = [-1, 0, 0, 0, 0]
                ra_range = int( 360 * 3600 * 1000 )
                ra_lo = int( ra_start * ( ra_range / index_ra_resolution ) )
                ra_hi = ra_lo + ra_range / index_ra_resolution

                if( index_file_offset == cached_index_data[0] ):
                    offset = cached_index_data[1]
                    end_offset = cached_index_data[2]
                else:
                    try:
                        index = open( self.indexfile, 'r' )
                        index.seek( index_file_offset, 0 )
                        offset, end_offset = np.fromstring( index.readline(), dtype = int, sep = ' ' )[0:2]
                        end_offset += offset
                        cached_index_data[0:3] = [index_file_offset, offset, end_offset]
                        index.close()
                    except Exception as e:
                        log.info( e + ", binary-search within entire zone" )
                        offset = 0
                        ifile.seek( 0, 2 )
                        end_offset = ifile.tell() / self.UCAC4_RAW.itemsize
                        ra_lo = 0
                        ra_hi = ra_range

                while( end_offset - offset > acceptable_limit ):
                    delta = end_offset - offset
                    minimum_bite = delta / 8 + 1
                    tval = delta * ( min_ra - ra_lo ) / ( ra_hi - ra_lo )
                    if( tval < minimum_bite ):
                        tval = minimum_bite
                    elif( tval > delta - minimum_bite ):
                        tval = delta - minimum_bite
                    toffset = offset + tval
                    ifile.seek( toffset * self.UCAC4_RAW.itemsize, 0 )
                    star = np.frombuffer( ifile.read( self.UCAC4_RAW.itemsize ), dtype = self.UCAC4_RAW )
                    if( star['ra'] < min_ra ):
                        offset = toffset
                        ra_lo = star[0]['ra']
                    else:
                        end_offset = toffset
                        ra_hi = star[0]['ra']

                ifile.seek( offset * self.UCAC4_RAW.itemsize, 0 )

                while 1:
                    stars = np.frombuffer( ifile.read( buffsize * self.UCAC4_RAW.itemsize ), dtype = self.UCAC4_RAW )
                    if len( stars ) == 0 or keep_going == 0:
                        break
                    for star in stars:
                        if( star['ra'] > max_ra ):
                            keep_going = 0
                            break
                        elif( star['ra'] > min_ra and
                                star['spd'] > min_spd and
                                star['spd'] < max_spd ):
                            rval += 1
                            tmpstars.append( star )
                            tmpids.append( [zone, offset + 1] )
                    
                        offset += 1

                ifile.close()

            zone += 1
        self._output_catalog( tmpstars, tmpids, keep = keep )
        if( rval >=0 and ra >= 0. and ra < 360. ):
            if( ra1 < 0. ):
                log.info( "Searching backwards." )
                rval += self.extract( ra + 360., dec, width, height, keep = 1 )
            if( ra2 > 360. ):
                log.info( "Searching forwards." )
                rval += self.extract( ra - 360., dec, width, height, keep = 1 )

        return( rval )




class UCAC5:

    def __init__( self, path = None ):
        if path == None:
            self.path = os.environ.get( "UCAC5_PATH" )
        else:
            self.path = path
        if self.path == None:
            log.error( "UCAC5 path is not set." )
            raise
        if len( glob.glob( os.path.join( self.path, "z???" ) ) ) == 0:
            log.error( "No UCAC5 binary file found." )
            self.valid = 0
            raise
        else:
            self.valid = 1
        self.indexfile = os.path.join( self.path, "u5index.asc" )
        if not os.path.isfile( self.indexfile ):
            log.warning( "No UCAC5 index file found." )
            self.indexfile = None
        self.data = Table()
        self.UCAC5_RAW = np.dtype( [ 
            ('gaiaid', np.int64),
            ('ra_g', np.int32), ('dec_g', np.int32),
            ('ra_g_e', np.int16), ('dec_g_e', np.int16),
            ('flag', np.int8), ('n', np.int8),
            ('epoch', np.int16),
            ('ra', np.int32), ('dec', np.int32),
            ('pm_ra', np.int16), ('pm_dec', np.int16),
            ('pm_ra_e', np.int16), ('pm_dec_e', np.int16),
            ('mag_g', np.int16), ('mag1', np.int16), ('mag_r', np.int16),
            ('mag_j', np.int16), ('mag_h', np.int16), ('mag_k', np.int16),
            ] )
        self.UCAC5_STAR = np.dtype( [ 
            ('gaiaid', np.int64),
            ('ra_g', np.float), ('dec_g', np.float),
            ('ra_g_e', np.float), ('dec_g_e', np.float),
            ('flag', np.int8), ('n', np.int8),
            ('epoch', np.float),
            ('ra', np.float), ('dec', np.float),
            ('pm_ra', np.float), ('pm_dec', np.float),
            ('pm_ra_e', np.float), ('pm_dec_e', np.float),
            ('mag_g', np.float), ('mag1', np.float), ('mag_r', np.float),
            ('mag_j', np.float), ('mag_h', np.float), ('mag_k', np.float),
            ] )


    def _get_ucac5_zone_file( self, zone_number, path ):
        filename = "z{0:03d}".format( zone_number )
        if os.path.isfile( os.path.join( path, filename ) ):
            return os.path.join( path, filename )
        else:
            return None

    def _get_index_file_offset( self, zone, ra_start ):
        rval = ( zone - 1 ) * ( 1440 * 21 + 6 ) + ra_start * 21
        if ( ra_start ):
            rval += 6
        return rval

    def _output_catalog( self, stars, ids, keep = 0 ):
        if stars == []:
            return
        stars = np.array( stars ).astype( self.UCAC5_STAR )
        stars['ra_g'] /= 3600000.
        stars['dec_g'] /= 3600000.
        stars['ra_g_e'] /= 10.
        stars['dec_g_e'] /= 10.
        stars['epoch'] /= 1000.
        stars['epoch'] += 1997.0
        stars['ra'] /= 3600000.
        stars['dec'] /= 3600000.
        stars['pm_ra'] /= 10.
        stars['pm_dec'] /= 10.
        stars['pm_ra_e'] /= 10.
        stars['pm_dec_e'] /= 10.
        stars['mag_g'] /= 1000.
        stars['mag1'] /= 1000.
        stars['mag_r'] /= 1000.
        stars['mag_j'] /= 1000.
        stars['mag_h'] /= 1000.
        stars['mag_k'] /= 1000.
        stars = Table( stars )
        ids = Table( np.array( ids ), names = ( 'zone', 'offset' ) )
        stars = hstack( [stars, ids] )

        if keep:
            if not isinstance( self.data, Table ):
                self.data = Table()
            self.data = vstack( [self.data, stars] )
        else:
            self.data = stars

    def extract( self, ra, dec, width, height, keep = 0 ):
        if not self.valid:
            return
        tmpstars = []
        tmpids = []
        rval = 0
        ra1 = ra - width / 2.
        ra2 = ra + width / 2.
        dec1 = dec - height / 2.
        dec2 = dec + height / 2.
        zone_height = .2
        zone = int( ( dec1 + 90. ) / zone_height ) + 1
        end_zone = int( ( dec2 + 90. ) / zone_height ) + 1
        index_ra_resolution = 1440
        ra_start = int( ra1 * index_ra_resolution / 360. )
        buffsize = 400
        if ( zone < 1 ):
            zone = 1
        if ( ra_start < 0 ):
            ra_start = 0

        while( rval >= 0 and zone <= end_zone ):
            try:
                ifile = open( self._get_ucac5_zone_file( zone, self.path ), 'rb' )
            except Exception as e:
                log.warning( "Zone file open failed," + e )
                ifile = None
            if ( ifile ):
                keep_going = 1
                max_ra = int( ra2 * 3600. * 1000. )
                min_ra = int( ra1 * 3600. * 1000. )
                min_dec = int( dec1 * 3600. * 1000. )
                max_dec = int( dec2 * 3600. * 1000. )
                acceptable_limit = 40
                index_file_offset = self._get_index_file_offset( zone, ra_start )
                cached_index_data = [-1, 0, 0, 0, 0]
                ra_range = int( 360 * 3600 * 1000 )
                ra_lo = int( ra_start * ( ra_range / index_ra_resolution ) )
                ra_hi = ra_lo + ra_range / index_ra_resolution

                if( index_file_offset == cached_index_data[0] ):
                    offset = cached_index_data[1]
                    end_offset = cached_index_data[2]
                else:
                    try:
                        index = open( self.indexfile, 'r' )
                        index.seek( index_file_offset, 0 )
                        offset, end_offset = np.fromstring( index.readline(), dtype = int, sep = ' ' )[0:2]
                        end_offset += offset
                        cached_index_data[0:3] = [index_file_offset, offset, end_offset]
                        index.close()
                    except Exception as e:
                        log.info( e + ", binary-search within entire zone" )
                        offset = 0
                        ifile.seek( 0, 2 )
                        end_offset = ifile.tell() / self.UCAC5_RAW.itemsize
                        ra_lo = 0
                        ra_hi = ra_range

                while( end_offset - offset > acceptable_limit ):
                    delta = end_offset - offset
                    minimum_bite = delta / 8 + 1
                    tval = delta * ( min_ra - ra_lo ) / ( ra_hi - ra_lo )
                    if( tval < minimum_bite ):
                        tval = minimum_bite
                    elif( tval > delta - minimum_bite ):
                        tval = delta - minimum_bite
                    toffset = offset + tval
                    ifile.seek( toffset * self.UCAC5_RAW.itemsize, 0 )
                    star = np.frombuffer( ifile.read( self.UCAC5_RAW.itemsize ), dtype = self.UCAC5_RAW )
                    if( star['ra'] < min_ra ):
                        offset = toffset
                        ra_lo = star[0]['ra']
                    else:
                        end_offset = toffset
                        ra_hi = star[0]['ra']

                ifile.seek( offset * self.UCAC5_RAW.itemsize, 0 )

                while 1:
                    stars = np.frombuffer( ifile.read( buffsize * self.UCAC5_RAW.itemsize ), dtype = self.UCAC5_RAW )
                    if len( stars ) == 0 or keep_going == 0:
                        break
                    for star in stars:
                        if( star['ra'] > max_ra ):
                            keep_going = 0
                            break
                        elif( star['ra'] > min_ra and
                                star['dec'] > min_dec and
                                star['dec'] < max_dec ):
                            rval += 1
                            tmpstars.append( star )
                            tmpids.append( [zone, offset + 1] )
                    
                        offset += 1

                ifile.close()

            zone += 1
        self._output_catalog( tmpstars, tmpids, keep = keep )
        if( rval >=0 and ra >= 0. and ra < 360. ):
            if( ra1 < 0. ):
                log.info( "Searching backwards." )
                rval += self.extract( ra + 360., dec, width, height, keep = 1 )
            if( ra2 > 360. ):
                log.info( "Searching forwards."  )
                rval += self.extract( ra - 360., dec, width, height, keep = 1 )

        return( rval )

if __name__ == "__main__":

    import time
    t = time.time()
    cat = UCAC4()
    print( cat.extract( 50, -16.3, 0.2, 0.15 ) )
    print( time.time() - t )
    print( cat.data[10] )
