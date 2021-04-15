# -*- coding: utf-8 -*-
#
# connections.py
#
# This file is part of DeNSE.
#
# Copyright (C) 2019 SeNEC Initiative
#
# DeNSE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# DeNSE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeNSE. If not, see <http://www.gnu.org/licenses/>.


""" Class and functions for graph creation """

from collections import defaultdict, OrderedDict

import numpy as np
from shapely.geometry import MultiPolygon

from .. import _pygrowth as _pg
from ..elements import Population
from ..units import um
from .graph import SpatialMultiNetwork, SpatialNetwork


__all__ = [
    "generate_network",
    "get_connections",
]


# ------------------------------ #
# Conection generation functions #
# ------------------------------ #

def generate_network(source_neurons=None, target_neurons=None,
                     method="intersections", spine_density=0.5/(um**2),
                     connection_probability=0.2, default_synaptic_strength=1.,
                     only_new_connections=False, autapse_allowed=False,
                     multigraph=False, **kwargs):
    r"""
    Create the graph based on the neuron shapes, the spine density, and a
    connection probability.

    The number of connection made will depend on the number of contacts between
    an axon and a dendrite.
    At each contact site, the number of potential synapses is computed as:

    .. math::

        n_{s, p} = \rho_s \cdot A_I

    with :math:`\rho_s` the `spine_density` and :math:`A_I` the intersection
    area.

    And the number of actual synapses is then:

    .. math::

        N_s = n_{s,p} \cdot p_c

    with :math:`p_c` the connection probability.

    Parameters
    ----------
    source_neurons : list of neurons, optional (default: all neurons)
        Neurons which will possess the pre-synaptic compartments of the
        connections (i.e. be connected through their axons).
    target_neurons : list of neurons, optional (default: all neurons)
        Neurons which will possess the post-synaptic compartments of the
        connections (i.e. be connected through their dendrites or soma)
    method : str, optional (default: "intersection")
        Method which use to generate synapses. Either "intersections" (synapses
        can be generated only when neurites overlap) or "spines" (neurites can
        be connected if they are closer than a certain distance
        `max_spine_length`).
    spine_density : float (quantity), optional (default: :math:`0.5 \\mu m^{-2}`)
        Number of spines per unit area, determines how many synapses are
        possible given an area of interaction.
    connection_probability : float, optional (default: 0.2)
        Probability of making a synapse for each spine/axon interaction which
        has been found geometrically.
    default_synaptic_strength : float, optional (default: 1.)
        Number caracterizing the default strength of a synapse. If `multigraph`
        is False, equivalent connections will always have a strength which is
        a multiple of this value.
    only_new_connections : bool, optional (default: False)
        If true, only the potential synapses that have been found during the
        last simulation run will be used; otherwise, all potential sites found
        since time 0 will be used.
    autapse_allowed : bool, optional (default: False)
        Whether connection from a neuron onto itself are generated if possible.
    multigraph : bool, optional (default: False)
        Whether the graph returned is simple (only one connection between each
        pair of neurons) or a multigraph (multiple connections can exist
        between every two neurons).
        If false, multiple connections which may exist between two neurons are
        merged into one equivalent connection with an increased synaptic
        strength and the average properties of the real connections (e.g. from
        three synapses of strength 1. and soma-to-soma distances
        :math:`120 \\mu m`, :math:`140 \\mu m`, and :math:`160 \\mu m`, one will
        get a single connection of strength 3. and of average length
        :math:`140 \\mu m`).
    **kwargs : optional arguments
        When using the "spines" `method`, an additional argument
        `max_spine_length` must be passed, specifying the maximum length
        at which neighboring neurites can still be connected through a spine
        (must be a dimensioned quantity, a length).
        If :func:`~dense.morphology.get_connections` has been called before,
        the network can be directly created from the returned `data` by passsing
        ``edges=edges, data=data`` in the call to ``generate_network``.

    See also
    --------
    :func:`~dense.morphology.get_connections`

    Details on the connection algorithms are available on the graph generation
    page in the user manual.
    """
    edges = kwargs.get("edges", None)
    data  = kwargs.get("data", None)

    if source_neurons is None:
        source_neurons = _pg.get_neurons(as_ints=True)

    if target_neurons is None:
        target_neurons = _pg.get_neurons(as_ints=True)

    if edges is None:
        edges, data = get_connections(
            source_neurons=source_neurons, target_neurons=target_neurons,
            method=method, spine_density=spine_density,
            connection_probability=connection_probability,
            only_new_connections=only_new_connections,
            autapse_allowed=autapse_allowed, **kwargs)

    population = None

    neurons = set(source_neurons)
    neurons.update(target_neurons)

    population  = Population.from_gids(neurons)
    num_neurons = len(neurons)

    NetClass = SpatialMultiNetwork if multigraph else SpatialNetwork

    shape = _pg.get_environment()
    unit  = "micrometer" if shape is None else shape.unit

    positions = np.array(
        [neuron.position.to(unit).magnitude for neuron in population])

    # test if there is a network to create
    if not neurons and shape is None:
        raise RuntimeError('Cannot create a network without any neurons '
                           'or environment.')

    network = NetClass(population=population, shape=shape,
                       positions=positions, multigraph=multigraph)

    num_synapses = len(edges)

    # for multigraphs we keep the positions as valid attributes
    if not multigraph:
        del data["position"]

    network.new_edges(edges, attributes=data,
                      unit_strength=default_synaptic_strength)

    return network


def get_connections_old(source_neurons=None, target_neurons=None,
                    method="intersections", spine_density=0.5/(um**2),
                    connection_probability=0.2, autapse_allowed=False,
                    **kwargs):
    """
    Obtain connection between `source_neurons` and `target_neurons` through
    a given method for synapse generation.
    The number of connection made will depend on the number of contacts between
    an axon and a dendrite.
    At each contact site, the number of potential synapses is computed as:
    .. math::
        n_{s, p} = \rho_s \cdot A_I
    with :math:`\rho_s` the `spine_density` and :math:`A_I` the intersection
    area.
    And the number of actual synapses is then:
    .. math::
        N_s = n_{s,p} \cdot p_c
    with :math:`p_c` the connection probability.
    Parameters
    ----------
    source_neurons : list of neurons, optional (default: all neurons)
        Neurons which will possess the pre-synaptic compartments of the
        connections (i.e. be connected through their axons)
    target_neurons : list of neurons, optional (default: all neurons)
        Neurons which will possess the post-synaptic compartments of the
        connections (i.e. be connected through their dendrites or soma)
    method : str, optional (default: "intersections")
        Method which use to generate synapses. Either "intersections" (synapses
        can be generated only when neurites overlap) or "spines" (neurites can
        be connected if they are closer than a certain distance
        `max_spine_length`).
    spine_density : float (quantity), optional (default: :math:`0.5 \mu m^{-2}`)
        Number of spines per unit area, determines how many synapses are made
        given an area of interaction.
    connection_probability : float, optional (default: 0.2)
        Probability of making a synapse for each spine/axon interaction which
        has been found geometrically.
    only_new_connections : bool, optional (default: False)
        If true, only the potential synapses that have been found during the
        last simulation run will be used; otherwise, all potential sites found
        since time 0 will be used.
    autapse_allowed : bool, optional (default: False)
        Whether connection from a neuron onto itself are generated if possible.
    **kwargs : optional arguments
        When using the "spines" `method`, an additional argument
        `max_spine_length` must be passed, specifying the maximum length
        at which neighboring neurites can still be connected through a spine
        (must be a dimensioned quantity, a length).

    Returns
    -------
    edges : list of edges of shape (e, 2)
        The edges created.
    data : dict
        Edge attributes, including

        - "position", array of shape (e, 2) containing the positions of the
          synapses
        - "distance", an approximation of the cable distance between the
          neurons, given by the sum of the distances between the somas and the
          synapse.
    """
    crossings_only, max_spine_length = None, None
    axon_buffer_radius = -1.

    if method == "intersections":
        crossings_only = True
    elif method == "spines":
        crossings_only = False

        if 'max_spine_length' in kwargs:
            axon_buffer_radius = kwargs["max_spine_length"].m_as("micrometer")
        else:
            raise AttributeError("`max_spine_length` must be passed if "
                                 "`method` is 'spines'.")
    else:
        raise ValueError("`method` must be either 'intersections' or 'spines'.")

    if source_neurons is None:
        source_neurons = _pg.get_neurons(as_ints=True)

    if target_neurons is None:
        target_neurons = _pg.get_neurons(as_ints=True)

    source_set  = set(source_neurons)
    target_set  = set(target_neurons)
    all_neurons = source_set.union(target_set)
    
    edges, positions, distances = [], [], []

    if all_neurons:
        syn_density = spine_density.m_as("1 / micrometer**2")

        axons, dendrites, somas = _pg._get_geom_skeleton(
            all_neurons, axon_buffer_radius=axon_buffer_radius, add_gc=False)

        somas = somas[:, :2]

        for i, (axon_gid, axon_polygon) in enumerate(axons.items()):
            if axon_gid in source_set:
                for j, (dend_gid, vd) in enumerate(dendrites.items()):
                    connection_allowed = (dend_gid != axon_gid or
                                          autapse_allowed)

                    if dend_gid in target_set and connection_allowed:
                        etuple = (axon_gid, dend_gid)
                        for d_polygon in vd:
                            if axon_polygon.intersects(d_polygon):
                                _get_synapses_intersection(
                                    axon_polygon, d_polygon, syn_density,
                                    somas, connection_probability, etuple, i, j,
                                    edges, positions, distances)

    data = {
        "position": positions,
        "distance": distances
    }

    return edges, data


# ------------------------------ #
# Python-level synapse formation #
# ------------------------------ #

def _get_synapses_intersection(axon_polygon, d_polygon, synapse_density, somas,
                               connection_probability, etuple, i, j, edges,
                               positions, distances):
    '''
    Tool fuction to find synapses of intersecting neurites
    '''
    intsct = axon_polygon.intersection(d_polygon)

    if not isinstance(intsct, MultiPolygon):
        intsct = [intsct]
    
    for poly in intsct:
        total        = poly.area * synapse_density * connection_probability
        rnd          = np.random.random()
        num_synapses = int(total) + (1 if (total - int(total)) < rnd else 0)

        if num_synapses > 0:
            s_soma = np.array(somas[i])
            t_soma = np.array(somas[j])
            pos    = poly.centroid
            dist   = np.linalg.norm(s_soma - pos) \
                    + np.linalg.norm(t_soma - pos)

            positions.extend([pos]*num_synapses)
            edges.extend([etuple]*num_synapses)
            distances.extend([dist]*num_synapses)


def get_connections(source_neurons=None, target_neurons=None,
                    method="intersections", spine_density=0.5/(um**2),
                    connection_probability=0.2, autapse_allowed=False,
                    **kwargs):
    """
    THIS IS TOO SLOW.
    Obtain connection between `source_neurons` and `target_neurons` through
    a given method for synapse generation.

    The number of connection made will depend on the number of contacts between
    an axon and a dendrite.
    At each contact site, the number of potential synapses is computed as:

    .. math::

        n_{s, p} = \rho_s \cdot A_I

    with :math:`\rho_s` the `spine_density` and :math:`A_I` the intersection
    area.

    And the number of actual synapses is then:

    .. math::

        N_s = n_{s,p} \cdot p_c

    with :math:`p_c` the connection probability.

    Parameters
    ----------
    source_neurons : list of neurons, optional (default: all neurons)
        Neurons which will possess the pre-synaptic compartments of the
        connections (i.e. be connected through their axons)
    target_neurons : list of neurons, optional (default: all neurons)
        Neurons which will possess the post-synaptic compartments of the
        connections (i.e. be connected through their dendrites or soma)
    method : str, optional (default: "intersections")
        Method which use to generate synapses. Either "intersections" (synapses
        can be generated only when neurites overlap) or "spines" (neurites can
        be connected if they are closer than a certain distance
        `max_spine_length`).
    spine_density : float (quantity), optional (default: :math:`0.5 \mu m^{-2}`)
        Number of spines per unit area, determines how many synapses are made
        given an area of interaction.
    connection_probability : float, optional (default: 0.2)
        Probability of making a synapse for each spine/axon interaction which
        has been found geometrically.
    only_new_connections : bool, optional (default: False)
        If true, only the potential synapses that have been found during the
        last simulation run will be used; otherwise, all potential sites found
        since time 0 will be used.
    autapse_allowed : bool, optional (default: False)
        Whether connection from a neuron onto itself are generated if possible.
    **kwargs : optional arguments
        When using the "spines" `method`, an additional argument
        `max_spine_length` must be passed, specifying the maximum length
        at which neighboring neurites can still be connected through a spine
        (must be a dimensioned quantity, a length).

    Returns
    -------
    edges : list of edges of shape (e, 2)
        The edges created.
    data : dict
        Edge attributes, including

        - "weight", the number of synaptic contacts associated to that edge
        - "position", array of shape (e, 2) containing the positions of the
          synapses
        - "distance", an approximation of the cable distance between the
          neurons, given by the sum of the distances between the somas and the
          synapse.
    """
    crossings_only, max_spine_length = None, None
    axon_buffer_radius = -1.

    if method == "intersections":
        crossings_only = True
    elif method == "spines":
        crossings_only = False

        if 'max_spine_length' in kwargs:
            axon_buffer_radius = kwargs["max_spine_length"].m_as("micrometer")
        else:
            raise AttributeError("`max_spine_length` must be passed if "
                                 "`method` is 'spines'.")
    else:
        raise ValueError("`method` must be either 'intersections' or 'spines'.")

    if source_neurons is None:
        source_neurons = _pg.get_neurons(as_ints=True)

    if target_neurons is None:
        target_neurons = _pg.get_neurons(as_ints=True)

    source_set  = set(source_neurons)
    target_set  = set(target_neurons)
    all_neurons = source_set.union(target_set)

    edges, positions, distances = [], [], []

    if all_neurons:
        syn_density = spine_density.m_as("1 / micrometer**2")

        return _pg._generate_synapses(
            source_set, target_set, syn_density, connection_probability,
            axon_buffer_radius, autapse_allowed)

    return [], {}

