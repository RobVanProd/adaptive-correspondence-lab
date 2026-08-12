import itertools
import math

import numpy as np

from adaptive_correspondence.particle_filter_bias import (
    ParticleFilterSpecification,
    count_compositions,
    exact_bayes_trajectory,
    exact_particle_filter_moments,
    grouped_transition_distribution,
    particle_filter_count_kernel,
    simulate_labeled_particle_filters,
)


def _fixture() -> ParticleFilterSpecification:
    return ParticleFilterSpecification(
        initial_belief=(0.5, 0.3, 0.2),
        true_transition=(
            (0.75, 0.20, 0.05),
            (0.10, 0.75, 0.15),
            (0.05, 0.25, 0.70),
        ),
        filter_transition=(
            (0.70, 0.25, 0.05),
            (0.15, 0.70, 0.15),
            (0.05, 0.30, 0.65),
        ),
        true_likelihoods=((0.8, 0.3, 0.1), (0.2, 0.7, 0.9)),
        filter_likelihoods=((0.7, 0.35, 0.15), (0.25, 0.65, 0.85)),
    )


def test_exact_bayes_filter_matches_direct_normalize_predict_steps() -> None:
    spec = _fixture()
    trajectory = exact_bayes_trajectory(spec)
    belief = np.asarray(spec.initial_belief, dtype=np.float64)
    transition = np.asarray(spec.true_transition, dtype=np.float64)
    expected = [belief.copy()]
    for likelihood in spec.true_likelihoods:
        belief = belief @ transition
        belief *= np.asarray(likelihood)
        belief /= np.sum(belief)
        expected.append(belief.copy())
    np.testing.assert_allclose(trajectory, expected, atol=2e-15, rtol=0.0)


def test_grouped_transition_matches_labeled_particle_enumeration() -> None:
    spec = _fixture()
    transition = np.asarray(spec.filter_transition, dtype=np.float64)
    source_counts = np.asarray([1, 1, 0], dtype=np.int64)
    exact = grouped_transition_distribution(source_counts, transition)
    direct = {state: 0.0 for state in count_compositions(2)}
    for destinations in itertools.product(range(3), repeat=2):
        probability = transition[0, destinations[0]] * transition[1, destinations[1]]
        counts = tuple(np.bincount(destinations, minlength=3))
        direct[counts] += probability
    assert abs(sum(exact.values()) - 1.0) < 2e-15
    for counts, probability in direct.items():
        np.testing.assert_allclose(exact[counts], probability, atol=2e-15, rtol=0.0)


def test_count_kernel_rows_are_probability_distributions() -> None:
    spec = _fixture()
    kernel = particle_filter_count_kernel(
        particle_count=3,
        transition=np.asarray(spec.filter_transition),
        likelihood=np.asarray(spec.filter_likelihoods[0]),
    )
    np.testing.assert_allclose(np.sum(kernel, axis=1), 1.0, atol=3e-14, rtol=0.0)
    assert np.min(kernel) >= 0.0


def test_exact_moments_match_bruteforce_labeled_paths_at_n2_t1() -> None:
    spec = ParticleFilterSpecification(
        initial_belief=(0.5, 0.3, 0.2),
        true_transition=((0.8, 0.15, 0.05), (0.1, 0.8, 0.1), (0.05, 0.2, 0.75)),
        filter_transition=((0.8, 0.15, 0.05), (0.1, 0.8, 0.1), (0.05, 0.2, 0.75)),
        true_likelihoods=((0.7, 0.4, 0.2),),
        filter_likelihoods=((0.7, 0.4, 0.2),),
    )
    result = exact_particle_filter_moments(spec, particle_count=2)
    mean = np.zeros(3)
    mass = 0.0
    prior = np.asarray(spec.initial_belief)
    transition = np.asarray(spec.filter_transition)
    likelihood = np.asarray(spec.filter_likelihoods[0])
    for initial in itertools.product(range(3), repeat=2):
        p_initial = math.prod(prior[state] for state in initial)
        for predicted in itertools.product(range(3), repeat=2):
            p_prediction = p_initial * math.prod(
                transition[source, target]
                for source, target in zip(initial, predicted, strict=True)
            )
            weights = likelihood[list(predicted)]
            probs = weights / np.sum(weights)
            for selected in itertools.product(range(2), repeat=2):
                probability = p_prediction * math.prod(probs[index] for index in selected)
                belief = np.bincount(
                    [predicted[index] for index in selected], minlength=3
                ) / 2
                mean += probability * belief
                mass += probability
    np.testing.assert_allclose(mass, 1.0, atol=4e-15, rtol=0.0)
    np.testing.assert_allclose(result.mean_belief, mean, atol=5e-15, rtol=0.0)


def test_individual_particle_simulator_approaches_exact_mean() -> None:
    spec = _fixture()
    exact = exact_particle_filter_moments(spec, particle_count=4)
    rng = np.random.Generator(np.random.PCG64(771))
    beliefs, terminal_particles = simulate_labeled_particle_filters(
        spec, particle_count=4, replications=200_000, rng=rng, batch_size=4096
    )
    assert beliefs.shape == (200_000, 3)
    assert terminal_particles.shape == (200_000, 4)
    np.testing.assert_allclose(np.mean(beliefs, axis=0), exact.mean_belief, atol=2e-3)
