from abcvoting.preferences import Profile as ApprovalProfile


class MultiIssueProfile:
    """
    Multi-issue profiles.

    Parameters
    ----------
        num_voters : int
            Number of voters in this profile.
    """

    def __init__(self, num_voters):
        if num_voters <= 0:
            raise ValueError(str(num_voters) + " is not a valid number of voters")

        self.voters = list(range(num_voters))
        self.issues = []

    def copy(self):
        new_profile = MultiIssueProfile(len(self.voters))
        for issue in self.issues:
            new_profile.add_issue([voter.approved for voter in issue])
        return new_profile

    def add_issue(self, approval_profile):
        if len(approval_profile) != len(self.voters):
            raise ValueError(
                "Number of voters in approval profile for new issue differs from"
                " number of voters in multi-issue profile."
            )
        cands = set()
        for s in approval_profile:
            cands.update(s)
        num_cand = max(max(cands) + 1, 2)  # at least two candidates
        issue = ApprovalProfile(num_cand)
        issue.add_voters(approval_profile)
        self.issues.append(issue)

    def satisfaction(self, voter, outcome):
        sat = 0
        for issue, profile in enumerate(self.issues):
            if not (0 <= outcome[issue] < profile.num_cand):
                raise ValueError(
                    f"outcome {outcome} not valid "
                    f"(num_cand for issue {issue} is {profile.num_cand})"
                )
            if outcome[issue] in profile[voter].approved:
                sat += 1
        return sat

    def average_number_of_approvals(self):
        return sum(
            sum(len(voter.approved) for voter in profile) / len(profile)
            for profile in self.issues
        ) / len(self.issues)
