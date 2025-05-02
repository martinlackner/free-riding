def _larger(x, y, min_number):
    assert len(x) == len(y), f"{x} {y}"
    diff = [x1 - y1 for x1, y1 in zip(x, y)]
    return sum(diff) > 0


def thiele(profile, weights):
    # sequential Thiele rules
    outcome = []

    satisfaction = [0] * len(profile.voters)
    for issue_index, issue in enumerate(profile.issues):
        issue = profile.issues[issue_index]

        # scores are marginal scores (=differences)
        max_marg_scores = [-1] * len(issue)
        winner = None
        for cand in issue.candidates:
            marg_scores = sorted(
                [
                    weights[satisfaction[vi]] if cand in voter.approved else 0
                    for vi, voter in enumerate(issue)
                ]
            )
            # print(f"issue {issue_index}, score {cand}: {sum(marg_scores)}")
            if _larger(marg_scores, max_marg_scores, min(weights)):
                max_marg_scores = marg_scores
                winner = cand

        outcome.append(winner)
        for vi, voter in enumerate(issue):
            if winner in voter.approved:
                satisfaction[vi] += 1

    return outcome


def owa(profile, weights):
    # sequential OWA rules
    outcome = []

    satisfaction = [0] * len(profile.voters)
    for issue_index, issue in enumerate(profile.issues):
        max_score = -1
        winner = None

        for cand in issue.candidates:
            updated_satisfaction = list(satisfaction)
            for vi, voter in enumerate(issue):
                if cand in voter.approved:
                    updated_satisfaction[vi] += 1
            voters_by_satisfaction = sorted(
                profile.voters, key=lambda vi: updated_satisfaction[vi]
            )
            assert len(set(voters_by_satisfaction)) == len(profile.voters)
            score = sum(
                weights[index] * updated_satisfaction[vi]
                for index, vi in enumerate(voters_by_satisfaction)
            )
            if score > max_score:
                max_score = score
                winner = cand

        outcome.append(winner)
        for vi, voter in enumerate(issue):
            if winner in voter.approved:
                satisfaction[vi] += 1

    return outcome
