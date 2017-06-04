#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import astropy
from astropy.table import Table
import os, sys
import glob

f = open( os.devnull, 'w' )
#sys.stderr = f

class URAT1:

    def __init__( self, path = None ):
        if path == None:
            self.path = os.environ.get( "URAT1_PATH" )
        else:
            self.path = path
        if self.path == None:
            print( "URAT1 path is not set.", file = sys.stderr )
            return
        if len( glob.glob( os.path.join( self.path, "z???" ) ) ) == 0:
            print( "No URAT1 binary file found.", file = sys.stderr )
            self.valid = 0
        else:
            self.valid = 1
        self.indexfile = os.path.join( self.path, "v1index.asc" )
        if not os.path.isfile( self.indexfile ):
            print( "No URAT1 index file found.", file = sys.stderr )
            self.indexfile = None
        self.data = Table()
        self.URAT1_RAW = np.dtype( [ 
            ('ra', np.int32), ('spd', np.int32),
            ('coord_e_s', np.int16), ('coord_e_m', np.int16),
            ('nst', np.int8), ('nsu', np.int8),
            ('epoch', np.int16),
            ('mag1', np.int16), ('mag_e', np.int16),
            ('nsm', np.int8), ('ref', np.int8),
            ('nit', np.int16), ('niu', np.int16),
            ('ngt', np.int8), ('ngu', np.int8),
            ('pm_ra', np.int16),('pm_dec', np.int16),
            ('pm_e', np.int16),
            ('mf2', np.int8), ('mfa', np.int8),
            ('twomass_id', np.int32),
            ('mag_j',np.int16), ('mag_h',np.int16), ('mag_k',np.int16), 
            ('mag_e_j', np.int16),
            ('mag_e_h', np.int16),
            ('mag_e_k', np.int16),
            ('cc_flag_j', np.int8),
            ('cc_flag_h', np.int8),
            ('cc_flag_k', np.int8),
            ('phq_j', np.int8), ('phq_h', np.int8), ('phq_k', np.int8),
            ('apass_mag_b', np.int16),
            ('apass_mag_v', np.int16),
            ('apass_mag_g', np.int16),
            ('apass_mag_r', np.int16),
            ('apass_mag_i', np.int16),
            ('apass_mag_e_b', np.int16),
            ('apass_mag_e_v', np.int16),
            ('apass_mag_e_g', np.int16),
            ('apass_mag_e_r', np.int16),
            ('apass_mag_e_i', np.int16),
            ('ann', np.int8), ('ano', np.int8)
            ] )
        self.URAT1_STAR = np.dtype( [ 
            ('ra', np.float), ('dec', np.float),
            ('coord_e_s', np.float), ('coord_e_m', np.float),
            ('nst', np.int8), ('nsu', np.int8),
            ('epoch', np.float),
            ('mag1', np.float), ('mag_e', np.float),
            ('nsm', np.int8), ('ref', np.int8),
            ('nit', np.int16), ('niu', np.int16),
            ('ngt', np.int8), ('ngu', np.int8),
            ('pm_ra', np.float),('pm_dec', np.float),
            ('pm_e', np.float),
            ('mf2', np.int8), ('mfa', np.int8),
            ('twomass_id', np.int32),
            ('mag_j',np.float), ('mag_h',np.float), ('mag_k',np.float), 
            ('mag_e_j', np.float),
            ('mag_e_h', np.float),
            ('mag_e_k', np.float),
            ('cc_flag_j', np.int8),
            ('cc_flag_h', np.int8),
            ('cc_flag_k', np.int8),
            ('phq_j', np.int8), ('phq_h', np.int8), ('phq_k', np.int8),
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
            ('ann', np.int8), ('ano', np.int8)
            ] )


    def _get_urat1_zone_file( self, zone_number, path ):
        filename = "z{0:03d}".format( zone_number )
        if os.path.isfile( os.path.join( path, filename ) ):
            return os.path.join( path, filename )
        else:
            return None

    def _get_index_file_offset( self, zone, ra_start ):
        rval = ( zone - 326 ) * ( 1440 * 38 + 6 ) + ra_start * 38
        if ( ra_start ):
            rval += 6
        return rval

    def _output_catalog( self, stars, ids, keep = 0 ):
        if stars == []:
            return
        stars = np.array( stars )
        ids = np.array( ids )
        try:
            from matplotlib.mlab import rec_append_fields
            stars = rec_append_fields( stars, 'zone', ids[:,0] )
            stars = rec_append_fields( stars, 'offset', ids[:,1] )
        except:
            from numpy.lib.recfunctions import append_fields
            stars = append_fields( stars, 'zone', ids[:,0], usemask = False )
            stars = append_fields( stars, 'offset', ids[:,1], usemask = False )
        names = list( stars.dtype.names )
        names[1] = 'dec'
        stars.dtype.names = names
        stars = stars.astype( self.URAT1_STAR )
        stars['ra'] /= 3600000.
        stars['dec'] -= 324000000.
        stars['dec'] /= 3600000.
        stars['coord_e_s'] /= 1.
        stars['coord_e_m'] /= 1.
        stars['epoch'] /= 1000.
        stars['epoch'] += 2000.
        stars['mag1'] /= 1000.
        stars['mag_e'] /= 1000.
        stars['pm_ra'] /= 10.
        stars['pm_dec'] /= 10.
        stars['pm_e'] /= 10.
        stars['mag_j'] /= 1000.
        stars['mag_h'] /= 1000.
        stars['mag_k'] /= 1000.
        stars['mag_e_j'] /= 1000.
        stars['mag_e_h'] /= 1000.
        stars['mag_e_k'] /= 1000.
        stars['apass_mag_b'] /= 1000.
        stars['apass_mag_v'] /= 1000.
        stars['apass_mag_g'] /= 1000.
        stars['apass_mag_r'] /= 1000.
        stars['apass_mag_i'] /= 1000.
        stars['apass_mag_e_b'] /= 1000.
        stars['apass_mag_e_v'] /= 1000.
        stars['apass_mag_e_g'] /= 1000.
        stars['apass_mag_e_r'] /= 1000.
        stars['apass_mag_e_i'] /= 1000.

        if keep:
            if not isinstance( self.data, Table ):
                self.data = Table()
            self.data = astropy.table.vstack( [self.data, newdata] )
            self.data.reset_index( drop = True, inplace = True )
        else:
            self.data = Table( data = stars )

    def extract_urat1_stars( self, ra, dec, width, height, keep = 0 ):
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
                ifile = open( self._get_urat1_zone_file( zone, self.path ), 'rb' )
            except Exception as e:
                print( "Zone file open failed,", e, file = sys.stderr )
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
                        offset, end_offset = np.fromstring( index.readline(), dtype = int, sep = ' ' )[2:4]
                        end_offset += offset
                        cached_index_data[0:3] = [index_file_offset, offset, end_offset]
                        index.close()
                    except Exception as e:
                        print( e, ", binary-search within entire zone", file = sys.stderr )
                        offset = 0
                        ifile.seek( 0, 2 )
                        end_offset = ifile.tell() / self.URAT1_RAW.itemsize
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
                    ifile.seek( toffset * self.URAT1_RAW.itemsize, 0 )
                    star = np.frombuffer( ifile.read( self.URAT1_RAW.itemsize ), dtype = self.URAT1_RAW )
                    if( star['ra'] < min_ra ):
                        offset = toffset
                        ra_lo = star[0]['ra']
                    else:
                        end_offset = toffset
                        ra_hi = star[0]['ra']

                ifile.seek( offset * self.URAT1_RAW.itemsize, 0 )

                while 1:
                    stars = np.frombuffer( ifile.read( buffsize * self.URAT1_RAW.itemsize ), dtype = self.URAT1_RAW )
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
                print( "Searching backwards.", file = sys.stderr )
                rval += self.extract_urat1_stars( ra + 360., dec, width, height, keep = 1 )
            if( ra2 > 360. ):
                print( "Searching forwards.", file = sys.stderr )
                rval += self.extract_urat1_stars( ra - 360., dec, width, height, keep = 1 )

        return( rval )

if __name__ == "__main__":

    import time
    t = time.time()
    cat = URAT1()
    print( cat.extract_urat1_stars( 50, 16.3, 2, 1.5 ) )
    print( time.time() - t )
    print( cat.data[95] )
