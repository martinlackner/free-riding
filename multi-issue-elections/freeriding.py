import os
from os import path

from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from abcvoting import generate
from profile import MultiIssueProfile
from abcvoting.preferences import Voter
import rules
import pickle

NUM_VOTERS_DEFAULT = 20
NUM_CAND_DEFAULT = 5
NUM_ISSUES_DEFAULT = 20
NUM_INSTANCES_DEFAULT = 2000
APPROVAL_RADIUS_DEFAULT = 1.5

THIELE_GRANULARITY = 4
PAV_INDEX = THIELE_GRANULARITY - 1


def data_filename_from_parameters(
    scenario,
    freerider_model,
    num_voters,
    num_cand,
    num_issues,
    num_instances,
    approval_radius,
    voter_model,
):
    return (
        f"../{scenario}-{freerider_model}-{num_issues}issues-{num_cand}alternatives-"
        f"{num_voters}voters-{num_instances}-instances"
        f"-{voter_model}-with-approvalradius-{approval_radius}.pickle"
    )

def datafile_exists_or_in_progress(data_filename):
    return path.exists(data_filename) or path.exists(data_filename+".lock")

def create_lock(data_filename):
    with open(data_filename+".lock", "w") as f:
        f.write("")

def computation_done(data_filename, data):
    assert not path.exists(data_filename)
    with open(data_filename, "wb") as f:
        pickle.dump(data, f)
    try:
        os.remove(data_filename + ".lock")
    except FileNotFoundError:
        pass
    print(f"{data_filename} done")


def _freeriding_possible_for_specific_voter_single_issue(
    profile, rule, weights, voter_index
):
    outcome = rule(profile, weights)
    num_freeriding = 0
    num_freeriding_pos = 0
    num_freeriding_neg = 0
    delta_satisfaction = 0
    satisfaction = sum(
        1
        for issue_index, winner in enumerate(outcome)
        if winner in profile.issues[issue_index][voter_index].approved
    )
    for issue_index, issue in enumerate(profile.issues):
        winner = outcome[issue_index]
        if winner not in issue[voter_index].approved:
            continue

        # voter is satisfied by this issue --> free-riding?
        orig_ballot = set(issue[voter_index].approved)
        freeriding_possible = False
        for cand in issue.candidates:
            if cand == winner:
                continue
            issue[voter_index] = Voter([cand])
            new_outcome = rule(profile, weights)
            if new_outcome[issue_index] == outcome[issue_index]:
                freeriding_possible = True
                break

        new_outcome = rule(profile, weights)
        issue[voter_index] = Voter(orig_ballot)
        if not freeriding_possible:
            continue

        assert winner in profile.issues[issue_index][voter_index].approved
        new_satisfaction = sum(
            1
            for _issue_index, _winner in enumerate(new_outcome)
            if _winner in profile.issues[_issue_index][voter_index].approved
        )
        if new_satisfaction > satisfaction:
            num_freeriding_pos += 1
        if new_satisfaction < satisfaction:
            num_freeriding_neg += 1
        delta_satisfaction += new_satisfaction - satisfaction
        num_freeriding += 1
    if num_freeriding_neg + num_freeriding_pos > 0:
        risk = num_freeriding_neg / (num_freeriding_neg + num_freeriding_pos)
    else:
        risk = None
    if num_freeriding == 0:
        assert delta_satisfaction == 0
    else:
        delta_satisfaction /= num_freeriding

    return {
        "freeriding": min(1, num_freeriding),
        "freeriding_pos": min(1, num_freeriding_pos),
        "freeriding_neg": min(1, num_freeriding_neg),
        "risk": risk,
        "satisfaction": satisfaction,
        "delta_satisfaction": delta_satisfaction,
    }


def freeriding_possible_for_specific_voter(
    profile, rule, weights, voter_index, freerider_model
):
    if freerider_model == "single":
        return _freeriding_possible_for_specific_voter_single_issue(
            profile, rule, weights, voter_index
        )
    elif freerider_model == "repeated":
        return _freeriding_possible_for_specific_voter_repeated(
            profile, rule, weights, voter_index
        )
    else:
        raise RuntimeError


def _freeriding_possible_for_specific_voter_repeated(
    profile, rule, weights, voter_index
):
    outcome = rule(profile, weights)
    satisfaction = sum(
        1
        for issue_index, winner in enumerate(outcome)
        if winner in profile.issues[issue_index][voter_index].approved
    )
    mod_profile = profile.copy()
    mod_rounds = []
    for issue_index, issue in enumerate(mod_profile.issues):
        winner = outcome[issue_index]
        if winner in issue[voter_index].approved:
            # voters in group are satisfied by this issue --> free-riding?
            issue[voter_index].approved.remove(winner)
            assert winner in profile.issues[issue_index][voter_index].approved
            new_outcome = rule(mod_profile, weights)
            if new_outcome[issue_index] != outcome[issue_index]:
                # free-riding not possible, winner changes
                # undo changes
                issue[voter_index].approved.add(winner)
            mod_rounds.append(issue_index)
    new_outcome = rule(mod_profile, weights)
    new_satisfaction = sum(
        1
        for issue_index, winner in enumerate(new_outcome)
        if winner in profile.issues[issue_index][voter_index].approved
    )
    num_freeriding = 0
    if outcome != new_outcome:
        num_freeriding = 1
    num_freeriding_pos = 0
    if new_satisfaction > satisfaction:
        num_freeriding_pos = 1
    num_freeriding_neg = 0
    if new_satisfaction < satisfaction:
        num_freeriding_neg = 1

    delta_satisfaction = new_satisfaction - satisfaction

    if num_freeriding_neg + num_freeriding_pos > 0:
        risk = num_freeriding_neg
    else:
        risk = None

    return {
        "freeriding": num_freeriding,
        "freeriding_pos": num_freeriding_pos,
        "freeriding_neg": num_freeriding_neg,
        "delta_satisfaction": delta_satisfaction,
        "satisfaction": satisfaction,
        "risk": risk,  # risk is 1 if delta_satisfaction is negative, 0 otherwise
    }


def plot_euclidean_profile(voter_points, cand_points, name):
    fig, ax = plt.subplots(dpi=600, figsize=(6, 6))

    # Scatter plots with distinct markers for readability in B/W
    ax.scatter(
        [x for x, y in cand_points],
        [y for x, y in cand_points],
        alpha=0.5,
        s=40,
        marker="o",
        color="black",
    )

    ax.scatter(
        [x for x, y in voter_points],
        [y for x, y in voter_points],
        alpha=0.5,
        s=40,
        marker="x",
        color="darkgreen",
    )

    # Ensure equal aspect ratio
    ax.set_aspect("equal", adjustable="box")

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set limits
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    fig.tight_layout()
    plt.savefig(f"../sample-profile-{name}.png", dpi=600, bbox_inches="tight")


def generate_multi_issue_profile(
    num_voters, num_cand, num_issues, appr_radius, voter_model, plot=False
):
    miprof = MultiIssueProfile(num_voters=num_voters)
    if voter_model == "2d_square":
        voter_points = [
            generate.random_point(
                generate.PointProbabilityDistribution(
                    "2d_square", center_point=[0.0, 0.0], width=2
                )
            )
            for _ in range(num_voters)
        ]
    elif voter_model == "many_groups":
        voter_points = []
        for center_x, center_y, count in [
            (0.5, 0.5, int(0.2 * num_voters)),
            (0.5, -0.5, int(0.2 * num_voters)),
            (-0.5, 0.5, int(0.2 * num_voters)),
            (-0.5, -0.5, num_voters - 3 * int(0.2 * num_voters)),
        ]:
            voter_points += [
                generate.random_point(
                    generate.PointProbabilityDistribution(
                        "2d_gaussian", center_point=[center_x, center_y], sigma=0.2
                    )
                )
                for _ in range(count)
            ]
        assert len(voter_points) == num_voters
    elif voter_model.startswith("unbalanced"):
        if voter_model == "unbalanced":
            small_group = 0.2
        else:
            small_group = float(voter_model.removeprefix("unbalanced"))
            assert small_group < 1
        voter_points = []
        for center_x, center_y, count in [
            (-0.5, -0.5, int(small_group * num_voters)),
            (0.5, 0.5, num_voters - int(small_group * num_voters)),
        ]:
            voter_points += [
                generate.random_point(
                    generate.PointProbabilityDistribution(
                        "2d_gaussian", center_point=[center_x, center_y], sigma=0.1
                    )
                )
                for _ in range(count)
            ]
        assert len(voter_points) == num_voters
    else:
        raise RuntimeError

    for _ in range(num_issues):
        cand_points = [
            generate.random_point(
                generate.PointProbabilityDistribution(
                    "2d_square", center_point=[0.0, 0.0], width=2
                )
            )
            for _ in range(num_cand)
        ]
        profile = generate.random_euclidean_threshold_profile(
            num_voters=num_voters,
            num_cand=num_cand,
            voter_prob_distribution=None,
            candidate_prob_distribution=None,
            threshold=appr_radius,
            voter_points=voter_points,
            candidate_points=cand_points,
        )
        miprof.add_issue([voter.approved for voter in profile])
    if plot:
        plot_euclidean_profile(
            voter_points,
            cand_points,
            name=f"{voter_model}-num_voters{num_voters}-num_cand{num_cand}",
        )
    return miprof


def run_specific_experiment(
    name,
    rule,
    weights,
    miprofs,
    designated_freeriders,
    freerider_model,
    show_progress=False,
):
    results = {}
    if show_progress:
        print(
            f"Running {name} ({len(weights)} different weight vectors, {len(miprofs)} profiles)"
        )
        iterator = tqdm(range(len(weights)))
    else:
        iterator = range(len(weights))

    for rule_index in iterator:
        results[rule_index] = {"delta_satisfaction_scores": defaultdict(list)}
        for miprof in miprofs:
            for freerider in designated_freeriders:
                _results = freeriding_possible_for_specific_voter(
                    miprof,
                    rule,
                    weights[rule_index],
                    freerider,
                    freerider_model=freerider_model,
                )
                results[rule_index]["delta_satisfaction_scores"][freerider].append(
                    _results["delta_satisfaction"]
                )
                if "risk_denominator" not in results[rule_index]:
                    results[rule_index]["risk_denominator"] = defaultdict(int)
                for key, value in _results.items():
                    if key not in results[rule_index]:
                        results[rule_index][key] = defaultdict(int)

                    if key == "risk":
                        if value is not None:
                            results[rule_index]["risk"][freerider] += value
                            results[rule_index]["risk_denominator"][freerider] += 1
                    else:
                        results[rule_index][key][freerider] += value

        # normalize results
        for key, value in results[rule_index].items():
            if key == "risk_denominator":
                continue
            if key == "delta_satisfaction_scores":
                continue
            for freerider in designated_freeriders:
                if key == "risk":
                    if results[rule_index]["risk"][freerider] != 0:
                        results[rule_index]["risk"][freerider] /= results[rule_index][
                            "risk_denominator"
                        ][freerider]
                    continue
                results[rule_index][key][freerider] /= len(miprofs)
    return results



def different_rules():
    print("different_rules")

    num_instances = NUM_INSTANCES_DEFAULT
    for num_voters, num_issues, num_cand, appr_radius, voter_model in [
        (
            NUM_VOTERS_DEFAULT,
            NUM_ISSUES_DEFAULT,
            NUM_CAND_DEFAULT,
            APPROVAL_RADIUS_DEFAULT,
            "2d_square",
        ),
        (NUM_VOTERS_DEFAULT, NUM_ISSUES_DEFAULT, NUM_CAND_DEFAULT, 1.2, "2d_square"),
        (NUM_VOTERS_DEFAULT, NUM_ISSUES_DEFAULT, NUM_CAND_DEFAULT, 2, "2d_square"),
        (
            NUM_VOTERS_DEFAULT,
            NUM_ISSUES_DEFAULT,
            NUM_CAND_DEFAULT,
            APPROVAL_RADIUS_DEFAULT,
            "many_groups",
        ),
        (
            NUM_VOTERS_DEFAULT,
            NUM_ISSUES_DEFAULT,
            NUM_CAND_DEFAULT,
            APPROVAL_RADIUS_DEFAULT,
            "unbalanced",
        ),
    ]:
        for freerider_model in ["single", "repeated"]:
            filename = data_filename_from_parameters(
                scenario="different_rules",
                freerider_model=freerider_model,
                num_voters=num_voters,
                num_cand=num_cand,
                num_issues=num_issues,
                num_instances=num_instances,
                approval_radius=appr_radius,
                voter_model=voter_model,
            )

            if datafile_exists_or_in_progress(filename):
                continue
            else:
                create_lock(filename)
                print(f"calculating {filename} ...")

            generate_multi_issue_profile(
                num_voters=num_voters,
                num_cand=num_cand,
                num_issues=num_issues,
                appr_radius=appr_radius,
                voter_model=voter_model,
                plot=True,
            )  # generate sample plot

            # results do not exist yet
            data = []

            xs_thiele = [x / THIELE_GRANULARITY for x in range(30)]
            thiele_weights = [
                [(i + 1) ** (-alpha) for i in range(num_issues)] for alpha in xs_thiele
            ]
            leximin_weights = [1 / ((num_issues * num_voters) ** i) for i in range(100)]
            xs_owa = list(range(num_voters))
            owa_weights = [
                [1] * (num_voters - i - 1) + leximin_weights[: i + 1] for i in xs_owa
            ]

            miprofs = [
                generate_multi_issue_profile(
                    num_voters=num_voters,
                    num_cand=num_cand,
                    num_issues=num_issues,
                    voter_model=voter_model,
                    appr_radius=appr_radius,
                )
                for _ in range(num_instances)
            ]
            # average number of approvals per voter
            print(
                "  average number of approvals per voter:",
                sum(inst.average_number_of_approvals() for inst in miprofs)
                / num_instances,
            )

            plot_type = "different_rules"
            for name, rule, weights, xs in [
                (
                    "Sequential Thiele methods",
                    rules.thiele,
                    thiele_weights,
                    xs_thiele,
                ),
                ("Sequential OWA methods", rules.owa, owa_weights, xs_owa),
            ]:
                data.append(
                    (
                        name,
                        rule,
                        xs,
                        run_specific_experiment(
                            name,
                            rule,
                            weights,
                            miprofs,
                            designated_freeriders=range(num_voters),
                            freerider_model=freerider_model,
                            show_progress=True,
                        ),
                        num_instances,
                        num_voters,
                        plot_type,
                    )
                )

            computation_done(filename, data)


def different_num_voters():
    plot_type = "different_num_voters"
    print(plot_type)

    num_instances = NUM_INSTANCES_DEFAULT
    num_issues = NUM_ISSUES_DEFAULT
    num_cand = NUM_CAND_DEFAULT
    num_voters_vals = list(range(5, 41))
    voter_model = "2d_square"
    used_rules = [
        ("Sequential Thiele methods", rules.thiele),
        ("Sequential OWA methods", rules.owa),
    ]
    freerider_model = "single"
    results = {name: list() for name, _ in used_rules}

    filename = data_filename_from_parameters(
        scenario="different_num_voters",
        freerider_model=freerider_model,
        num_voters="variable-",
        num_cand=num_cand,
        num_issues=num_issues,
        num_instances=num_instances,
        approval_radius=APPROVAL_RADIUS_DEFAULT,
        voter_model=voter_model,
    )
    if datafile_exists_or_in_progress(filename):
        return
    else:
        create_lock(filename)
        print(f"calculating {filename} ...")

    data = []  # results do not exist yet

    for num_voters in tqdm(num_voters_vals):
        pav_weights = [1 / (i + 1) for i in range(num_issues)]
        leximin_weights = [1 / ((num_issues * num_voters) ** i) for i in range(100)]

        miprofs = [
            generate_multi_issue_profile(
                num_voters=num_voters,
                num_cand=num_cand,
                num_issues=num_issues,
                appr_radius=APPROVAL_RADIUS_DEFAULT,
                voter_model=voter_model,
            )
            for _ in range(num_instances)
        ]

        for name, rule in used_rules:
            if name == "Sequential Thiele methods":
                results[name].append(
                    run_specific_experiment(
                        name,
                        rule,
                        [pav_weights],
                        miprofs,
                        designated_freeriders=range(num_voters),
                        freerider_model=freerider_model,
                    )
                )
            elif name == "Sequential OWA methods":
                results[name].append(
                    run_specific_experiment(
                        name,
                        rule,
                        [leximin_weights],
                        miprofs,
                        designated_freeriders=range(num_voters),
                        freerider_model=freerider_model,
                    )
                )
            else:
                raise RuntimeError

    for name, rule in used_rules:
        data.append(
            (
                name,
                rule,
                num_voters_vals,
                results[name],
                num_instances,
                None,
                plot_type,
            )
        )

    computation_done(filename, data)


def different_num_cand():
    print("different_num_cand")

    num_instances = NUM_INSTANCES_DEFAULT
    num_issues = NUM_ISSUES_DEFAULT
    num_cand_vals = list(range(2, 10))
    num_voters = NUM_VOTERS_DEFAULT
    voter_model = "2d_square"
    used_rules = [
        ("Sequential Thiele methods", rules.thiele),
        ("Sequential OWA methods", rules.owa),
    ]
    freerider_model = "single"

    results = {name: list() for name, _ in used_rules}

    filename = data_filename_from_parameters(
        scenario="different_num_cand",
        freerider_model=freerider_model,
        num_voters=num_voters,
        num_cand="variable-",
        num_issues=num_issues,
        num_instances=num_instances,
        approval_radius=APPROVAL_RADIUS_DEFAULT,
        voter_model=voter_model,
    )
    if datafile_exists_or_in_progress(filename):
        return
    else:
        create_lock(filename)
        print(f"calculating {filename} ...")

    for num_cand in tqdm(num_cand_vals):
        pav_weights = [1 / (i + 1) for i in range(num_issues)]
        leximin_weights = [1 / ((num_issues * num_voters) ** i) for i in range(100)]

        miprofs = [
            generate_multi_issue_profile(
                num_voters=num_voters,
                num_cand=num_cand,
                num_issues=num_issues,
                appr_radius=APPROVAL_RADIUS_DEFAULT,
                voter_model="2d_square",
            )
            for _ in range(num_instances)
        ]

        for name, rule in used_rules:
            if name == "Sequential Thiele methods":
                results[name].append(
                    run_specific_experiment(
                        name,
                        rule,
                        [pav_weights],
                        miprofs,
                        designated_freeriders=range(num_voters),
                        freerider_model=freerider_model,
                    )
                )
            elif name == "Sequential OWA methods":
                results[name].append(
                    run_specific_experiment(
                        name,
                        rule,
                        [leximin_weights],
                        miprofs,
                        designated_freeriders=range(num_voters),
                        freerider_model=freerider_model,
                    )
                )
            else:
                raise RuntimeError

    plot_type = "different_num_cand"
    for name, rule in used_rules:
        assert len(results[name]) == len(num_cand_vals)
    data = [
        (
            name,
            rule,
            num_cand_vals,
            results[name],
            num_instances,
            num_voters,
            plot_type,
        )
        for name, rule in used_rules
    ]

    computation_done(filename, data)


def different_num_issues():
    print("different_num_issues")

    num_instances = NUM_INSTANCES_DEFAULT
    num_issues_vals = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    num_cand = NUM_CAND_DEFAULT
    num_voters = NUM_VOTERS_DEFAULT
    voter_model = "2d_square"
    used_rules = [
        ("Sequential Thiele methods", rules.thiele),
        ("Sequential OWA methods", rules.owa),
    ]
    freerider_model = "single"

    results = {name: list() for name, _ in used_rules}

    filename = data_filename_from_parameters(
        scenario="different_num_issues",
        freerider_model=freerider_model,
        num_voters=num_voters,
        num_cand=num_cand,
        num_issues="variable-",
        num_instances=num_instances,
        approval_radius=APPROVAL_RADIUS_DEFAULT,
        voter_model=voter_model,
    )
    if datafile_exists_or_in_progress(filename):
        return
    else:
        create_lock(filename)
        print(f"calculating {filename} ...")

    for num_issues in tqdm(num_issues_vals):
        pav_weights = [1 / (i + 1) for i in range(num_issues)]
        leximin_weights = [1 / ((num_issues * num_voters) ** i) for i in range(100)]

        miprofs = [
            generate_multi_issue_profile(
                num_voters=num_voters,
                num_cand=num_cand,
                num_issues=num_issues,
                appr_radius=APPROVAL_RADIUS_DEFAULT,
                voter_model=voter_model,
            )
            for _ in range(num_instances)
        ]

        for name, rule in used_rules:
            if name == "Sequential Thiele methods":
                results[name].append(
                    run_specific_experiment(
                        name,
                        rule,
                        [pav_weights],
                        miprofs,
                        designated_freeriders=range(num_voters),
                        freerider_model=freerider_model,
                    )
                )
            elif name == "Sequential OWA methods":
                results[name].append(
                    run_specific_experiment(
                        name,
                        rule,
                        [leximin_weights],
                        miprofs,
                        designated_freeriders=range(num_voters),
                        freerider_model=freerider_model,
                    )
                )
            else:
                raise RuntimeError

    plot_type = "different_num_issues"
    data = [
        (
            name,
            rule,
            num_issues_vals,
            results[name],
            num_instances,
            num_voters,
            plot_type,
        )
        for name, rule in used_rules
    ]

    computation_done(filename, data)


def designated_freerider():
    print("designated_freerider")

    num_instances = NUM_INSTANCES_DEFAULT
    # in these experiments 40 voters!
    for num_voters, num_issues, num_cand, appr_radius, voter_model in [
        (
            NUM_VOTERS_DEFAULT,
            NUM_ISSUES_DEFAULT,
            NUM_CAND_DEFAULT,
            APPROVAL_RADIUS_DEFAULT,
            "2d_square",
        ),
        (
            NUM_VOTERS_DEFAULT,
            NUM_ISSUES_DEFAULT,
            NUM_CAND_DEFAULT,
            APPROVAL_RADIUS_DEFAULT,
            "many_groups",
        ),
        (
            NUM_VOTERS_DEFAULT,
            NUM_ISSUES_DEFAULT,
            NUM_CAND_DEFAULT,
            APPROVAL_RADIUS_DEFAULT,
            "unbalanced",
        ),
        (
            NUM_VOTERS_DEFAULT,
            NUM_ISSUES_DEFAULT,
            NUM_CAND_DEFAULT,
            APPROVAL_RADIUS_DEFAULT,
            "unbalanced0.3",
        ),
    ]:
        for freerider_model in ["single", "repeated"]:
            filename = data_filename_from_parameters(
                scenario="designated_freerider",
                freerider_model=freerider_model,
                num_voters=num_voters,
                num_cand=num_cand,
                num_issues=num_issues,
                num_instances=num_instances,
                approval_radius=appr_radius,
                voter_model=voter_model,
            )

            if datafile_exists_or_in_progress(filename):
                continue
            else:
                create_lock(filename)
                print(f"calculating {filename} ...")

            generate_multi_issue_profile(
                num_voters=num_voters,
                num_cand=num_cand,
                num_issues=num_issues,
                appr_radius=appr_radius,
                voter_model=voter_model,
                plot=True,
            )  # generate sample plot

            # results do not exist yet
            data = []

            xs_thiele = [x / THIELE_GRANULARITY for x in range(30)]
            thiele_weights = [
                [(i + 1) ** (-alpha) for i in range(num_issues)] for alpha in xs_thiele
            ]
            leximin_weights = [1 / ((num_issues * num_voters) ** i) for i in range(100)]
            xs_owa = list(range(num_voters))
            owa_weights = [
                [1] * (num_voters - i - 1) + leximin_weights[: i + 1] for i in xs_owa
            ]

            miprofs = [
                generate_multi_issue_profile(
                    num_voters=num_voters,
                    num_cand=num_cand,
                    num_issues=num_issues,
                    voter_model=voter_model,
                    appr_radius=appr_radius,
                )
                for _ in range(num_instances)
            ]

            plot_type = "designated_freerider"
            for name, rule, weights, xs in [
                (
                    "Sequential Thiele methods",
                    rules.thiele,
                    thiele_weights,
                    xs_thiele,
                ),
                ("Sequential OWA methods", rules.owa, owa_weights, xs_owa),
            ]:
                data.append(
                    (
                        name,
                        rule,
                        xs,
                        run_specific_experiment(
                            name,
                            rule,
                            weights,
                            miprofs,
                            designated_freeriders=[0, 19],
                            freerider_model=freerider_model,
                            show_progress=True,
                        ),
                        num_instances,
                        num_voters,
                        plot_type,
                    )
                )

            computation_done(filename, data)


if __name__ == "__main__":
    designated_freerider()
    different_rules()
    different_num_issues()
    different_num_cand()
    different_num_voters()
