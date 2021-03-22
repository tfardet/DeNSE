/*
 * steering_memory_based.cpp
 *
 * This file is part of DeNSE.
 *
 * Copyright (C) 2019 SeNEC Initiative
 *
 * DeNSE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * DeNSE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DeNSE. If not, see <http://www.gnu.org/licenses/>.
 */

#include "steering_memory_based.hpp"

#include <cmath>

#include "config.hpp"

#include "GrowthCone.hpp"


namespace growth
{

MemBasedSteeringModel::MemBasedSteeringModel(GCPtr gc, NeuritePtr neurite)
    : SteeringModel(gc, neurite)
    , memory_angle_(fmod(gc->get_state("angle"), 2 * M_PI))
    , memory_persistence_(0.9)
    , memory_influence_(1.)
{}


MemBasedSteeringModel::MemBasedSteeringModel(const MemBasedSteeringModel &copy,
                                             GCPtr gc, NeuritePtr neurite)
    : SteeringModel(copy, gc, neurite)
    , memory_angle_(fmod(gc->get_state("angle"), 2 * M_PI))
    , memory_persistence_(copy.memory_persistence_)
    , memory_influence_(copy.memory_influence_)
{}


void MemBasedSteeringModel::compute_direction_probabilities(
    std::vector<double> &directions_weights, const Filopodia &filo,
    double substep, double &total_proba, bool &stuck)
{
    double weight, angle, previous_angle, dtheta, d1, d2, dmax, amin, amax;
    bool in_progress = true;
    unsigned int n2;

    stuck       = true;
    total_proba = 0.;

    double current_angle =
        fmod(gc_weakptr_.lock()->get_state("angle"), 2 * M_PI);

    // update memory angle:
    double afactor = substep*(1 - memory_persistence_);

    memory_angle_ = fmod(
        (afactor*current_angle + memory_persistence_ * memory_angle_)
        / (memory_persistence_ + afactor), 2*M_PI);

    // loop over the angles, get total probability, and add memory contribution
    stype n_max  = filo.directions.size() - 1;

    dtheta = 0.5 * (filo.directions[1] - filo.directions[0]);

    dmax = M_PI + filo.directions[0];

    amin = fmod(filo.directions[0] + current_angle, 2*M_PI);
    amax = fmod(filo.directions[n_max] + current_angle, 2*M_PI);

    // todo: first loop and get all angles that are not NaN

    std::vector<unsigned int> valid_directions;
    std::vector<double> distances;

    for (unsigned int n = 0; n < filo.directions.size(); n++)
    {
        weight = directions_weights[n];

        if (not std::isnan(weight))
        {
            total_proba += weight;
            stuck = false;

            // store index and compute angluar distance to memory angle
            valid_directions.push_back(n);

            angle = filo.directions[n];

            distances.push_back(
                std::abs(fmod(angle - memory_angle_ + M_PI, 2*M_PI) - M_PI));
        }
    }

    // then: find the closest angle
    unsigned int chosen =
        std::min_element(distances.begin(), distances.end())
        - distances.begin();

    directions_weights[valid_directions[chosen]] += memory_influence_;
}


void MemBasedSteeringModel::set_status(const statusMap &status)
{
    double mp, mi;
    bool b;

    b = get_param(status, names::memory_persistence, mp);
    if (b)
    {
        if (mp < 0 or mp > 1)
        {
            throw std::invalid_argument("`" + names::memory_persistence +
                                        "` must be in [0, 1].");
        }

        memory_persistence_ = mp;
    }

    b = get_param(status, names::memory_influence, mi);
    if (b)
    {
        if (mi < 0)
        {
            throw std::invalid_argument("`" + names::memory_influence +
                                        "` must be positive.");
        }

        memory_influence_ = mi;
    }
}


void MemBasedSteeringModel::get_status(statusMap &status) const
{
    set_param(status, names::memory_persistence, memory_persistence_, "");
    set_param(status, names::memory_influence, memory_influence_, "");
}

} // namespace growth
