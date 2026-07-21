import itertools
import torch
import torch.nn.functional as F


def si_sdr(estimate, target, eps=1e-8):
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)

    projection = (
        torch.sum(
            estimate * target,
            dim=-1,
            keepdim=True
        )
        /
        (
            torch.sum(
                target ** 2,
                dim=-1,
                keepdim=True
            )
            + eps
        )
    ) * target

    noise = estimate - projection

    ratio = (
        torch.sum(
            projection ** 2,
            dim=-1
        )
        /
        (
            torch.sum(
                noise ** 2,
                dim=-1
            )
            + eps
        )
    )

    return 10 * torch.log10(
        ratio + eps
    )


def pairwise_si_sdr(estimates, targets):
    num_sources = estimates.shape[1]

    scores = torch.zeros(
        estimates.shape[0],
        num_sources,
        num_sources,
        device=estimates.device
    )

    for i in range(num_sources):
        for j in range(num_sources):
            scores[:, i, j] = si_sdr(
                estimates[:, i],
                targets[:, j]
            )

    return scores


def pit_si_sdr_loss(estimates, targets):
    batch_size = estimates.shape[0]
    num_sources = estimates.shape[1]

    scores = pairwise_si_sdr(
        estimates,
        targets
    )

    perms = list(
        itertools.permutations(
            range(num_sources)
        )
    )

    best_scores = []

    for b in range(batch_size):
        perm_scores = []

        for perm in perms:
            score = 0

            for i in range(num_sources):
                score += scores[
                    b,
                    i,
                    perm[i]
                ]

            perm_scores.append(score)

        best_scores.append(
            torch.max(
                torch.stack(
                    perm_scores
                )
            )
        )

    best_scores = torch.stack(
        best_scores
    )

    loss = -torch.mean(
        best_scores
    )

    return loss


def si_sdr_improvement(
    estimate,
    mixture,
    target
):
    estimate_score = si_sdr(
        estimate,
        target
    )

    mixture_score = si_sdr(
        mixture,
        target
    )

    return (
        estimate_score -
        mixture_score
    )