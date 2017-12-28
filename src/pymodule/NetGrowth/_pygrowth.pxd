#!/usr/bin/env cython
#-*- coding:utf-8 -*-

""" Generation tools for NNGT """

from libc.stdint cimport uintptr_t

from libcpp cimport bool

from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector


# ---------------------- #
# Load the c++ functions #
# ---------------------- #

cdef extern from "../libgrowth/elements_types.hpp" namespace "growth":
    ctypedef pair[vector[double], vector[double]] SkelNeurite
    ctypedef vector[vector[double]] SkelSomas


cdef extern from "../libgrowth/config.hpp" namespace "growth":
    ctypedef enum dtype:
        BOOL, DOUBLE, INT, SIZE, VEC_SIZE, VEC_LONG, STRING, VEC_STRING

    ctypedef unordered_map[string, Property] statusMap

    cdef cppclass Property:
        Property()
        Property(double d)
        Property(int i)
        Property(size_t ul)
        Property(const vector[size_t]& v)
        Property(const vector[long]& v)
        Property(const string& s)
        Property(const vector[string]& v)
        dtype data_type
        bool b
        double d
        int i
        size_t ul
        vector[long] ll
        vector[size_t] uu
        string s
        vector[string] ss


cdef extern from "../libgrowth/growth_time.hpp" namespace "growth":
    cdef cppclass CTime "growth::Time":
        CTime()
        CTime(float seconds, unsigned char minutes, unsigned char hours,
             unsigned char days)
        CTime(CTime initial_time, unsigned long steps)
        void set_sec(float seconds)
        void set_min(char minutes)
        void set_hour(char hours)
        void set_day(char days)
        float get_sec() const
        char get_min() const
        char get_hour() const
        char get_day() const
        double get_total_seconds() const


# exception
cdef extern from "../libgrowth/exceptions.hpp" namespace "growth":
    cdef cppclass InvalidArg:
        InvalidArg(const string&, const char*, const char*, unsigned int)

    cdef cppclass BadPropertyType:
        BadPropertyType(const string&, const char*, const char*,
                        unsigned int)

    cdef cppclass BadPropertyName:
        BadPropertyType(const string&, const string&, const string&,
                        const char*, const char*, unsigned int)


# kernel functions
ctypedef unordered_map[ string, vector[double] ] mapParams


cdef extern from "../module.hpp" namespace "growth":
    cdef void init_growth( int* argc, char** argv[] ) except +

    cdef void finalize_growth() except +

    cdef size_t create_objects(const string& object_name,
                               const vector[statusMap]& obj_params) except +

    cdef size_t create_neurons(const vector[statusMap]& neuron_params,
                               const vector[statusMap]& axon_params,
                               const vector[statusMap]& dendrites_params
                               ) except +

    cdef void get_environment ( GEOSGeometry* &environment ) except +

    cdef void set_environment ( GEOSGeometry* environment ) except +

    cdef const CTime get_current_time() except +

    cdef statusMap get_kernel_status() except +

    cdef size_t get_num_objects() except +

    cdef void get_skeleton(
        SkelNeurite& axon, SkelNeurite& dendrites, SkelNeurite& nodes,
        SkelNeurite& growth_cones, SkelSomas& somas,
        vector[size_t] gids) except +

    cdef void get_swc(string output_file,
        vector[size_t] gids, unsigned int resolution) except +

    cdef statusMap get_status(size_t gid) except +

    cdef statusMap get_neurite_status(size_t gid,
                                      const string& n_type) except +

    cdef vector[size_t] get_neurons() except +

    cdef void get_defaults(const string& object_name,
                           const string& object_type,
                           statusMap &status) except +

    cdef void get_models(vector[string]& models,
                         const string& object_type) except +

    cdef void get_recorder_type(size_t gid, string& level,
                                string& event_type) except +

    cdef bool get_next_recording(size_t gid, vector[Property]& ids,
                                 vector[double]& values) except +

    cdef bool get_next_time(size_t gid, vector[Property]& ids,
                            vector[double]& values,
                            const string& time_units) except +

    cdef string object_type(size_t gid) except +

    cdef void reset_kernel() except +

    cdef void set_kernel_status(statusMap status_dict,
                                string c_simulation_ID) except +

    cdef string get_simulation_ID() except +

    cdef void set_status(size_t gid, statusMap neuron_status,
                         statusMap axon_status,
                         statusMap dendrites_status) except +

    cdef void simulate(const CTime& simtime) except +

    cdef void test_random_generator(vector[vector[double]]& values,
                                    size_t size) except+


# ---------------------- #
# GEOS-related functions #
# ---------------------- #

cdef extern from "geos_c.h":
    ctypedef struct GEOSGeometry


cdef inline GEOSGeometry *geos_from_shapely(shapely_geom) except *:
    '''
    Get the GEOS geometry pointer from the given shapely geometry.
    '''
    cdef uintptr_t geos_geom = shapely_geom._geom
    return <GEOSGeometry *>geos_geom
