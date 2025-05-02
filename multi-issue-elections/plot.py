import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import rules
import os
import seaborn as sns
import pandas as pd
from freeriding import PAV_INDEX, APPROVAL_RADIUS_DEFAULT
from pathlib import Path
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def save_fig(_plt, filepath):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    for filetype in ["pdf", "png"]:
        plot_fp = f"{filepath}.{filetype}"
        if plot_fp in written_files:
            raise RuntimeError(f"{plot_fp} was already written.")
        else:
            written_files.add(plot_fp)
        _plt.savefig(plot_fp, dpi=600, bbox_inches="tight")


def distribution_name(filename):
    if "2d_square" in filename:
        distribution = "square"
    elif "many_groups" in filename:
        distribution = "many groups"
    elif "unbalanced" in filename:
        distribution = "unbalanced"
    else:
        raise RuntimeError
    return distribution


def plot_single_freeriders(data):
    print(f"{filename} ...")
    for ruleclass, rule, xs, results, num_instances, num_voters, plot_type in data:
        # plot
        fig, ax = plt.subplots(figsize=(8, 4))
        freeriding_pos = [
            np.mean(
                [results[rule_index]["freeriding_pos"][vi] for vi in range(num_voters)]
            )
            for rule_index in range(len(xs))
        ]
        ax.plot(
            xs,
            freeriding_pos,
            "g--",
            label="likelihood of successful free-riding",
        )
        freeriding_neg = [
            np.mean(
                [results[rule_index]["freeriding_neg"][vi] for vi in range(num_voters)]
            )
            for rule_index in range(len(xs))
        ]
        ax.plot(
            xs,
            freeriding_neg,
            "r-",
            label="likelihood of harmful free-riding",
        )
        risk = [
            np.mean([results[rule_index]["risk"][vi] for vi in range(num_voters)])
            for rule_index in range(len(xs))
        ]
        ax.plot(xs, risk, "k-.", label="risk")
        # plt.xlim(-.02)

        plt.title(f"{ruleclass}, {distribution_name(filename)} distribution")
        # plt.ylabel("Proportion of voters (average)")
        ax.grid(visible=True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if plot_type == "different_rules":
            if "-single-" in filename:
                plt.ylim(-0.02, 0.66)
            elif "-repeated-" in filename:
                plt.ylim(-0.02, 0.96)
            else:
                raise RuntimeError

            plt.xlabel("Parameter x")
            if rule == rules.thiele:
                ticks = list(range(8))
                dic = {0: "0\nutilit.", 1: "1\nPAV"}
            elif rule == rules.owa:
                ticks = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
                dic = {0: "0\nutilit.", 19: f"{19}\nleximin"}
            else:
                raise RuntimeError
        # elif plot_type in [
        #     "different_num_voters",
        #     "different_num_cand",
        #     "different_num_issues",
        # ]:
        #     plt.ylim(-0.02, 0.66)
        #     ticks = xs
        #     dic = {}
        #     plt.xlabel(plot_type.removeprefix("different_num_"))
        else:
            raise RuntimeError

        labels = [
            ticks[i] if t not in dic.keys() else dic[t] for i, t in enumerate(ticks)
        ]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

        plt.legend(loc="upper left")
        save_fig(
            plt,
            f"../{plot_type}/{ruleclass.replace(" ", "-")}"
            f"-{extra_info_from_filename(filename)}",
        )
        plt.close()

        # extra info
        if rule == rules.thiele:
            print(f" - PAV {distribution_name(filename)} risk={risk[PAV_INDEX]}")
        elif rule == rules.owa:
            print(
                f" - leximin {distribution_name(filename)} risk={risk[num_voters - 1]}"
            )
    print(f"{filename} done.")


def extra_info_from_filename(filename):
    return filename.removesuffix(".pickle").replace(" ", "-").replace(".", "dot")


def plot_designated_freerider(data):
    for ruleclass, rule, xs, results, num_instances, num_voters, plot_type in data:
        fig, ax = plt.subplots(figsize=(8, 4))

        for rule_index in range(len(xs)):
            assert list(results[rule_index]["delta_satisfaction"].keys()) == [
                0,
                num_voters - 1,
            ]
        ax.plot(
            xs,
            [
                results[rule_index]["delta_satisfaction"][0]
                for rule_index in range(len(xs))
            ],
            "r-",
            label="minority freerider: change in satisfaction",
        )
        ax.plot(
            xs,
            [
                results[rule_index]["delta_satisfaction"][num_voters - 1]
                for rule_index in range(len(xs))
            ],
            "g--",
            label="majority freerider: change in satisfaction",
        )
        # plt.xlim(-.02)
        plt.title(f"{ruleclass}, {distribution_name(filename)} distribution")
        # plt.ylabel("Proportion of voters (average)")
        ax.grid(visible=True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # plt.ylim(-0.02, 0.66)
        plt.xlabel("Parameter x")
        if rule == rules.thiele:
            ticks = list(range(8))
            dic = {0: "0\nutilit.", 1: "1\nPAV"}
        elif rule == rules.owa:
            ticks = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
            dic = {0: "0\nutilit.", 19: f"{19}\nleximin"}
        else:
            raise RuntimeError

        labels = [
            ticks[i] if t not in dic.keys() else dic[t] for i, t in enumerate(ticks)
        ]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

        # ax.axvline(1, ls="--")
        # plt.text(20,0,'PAV',rotation=90)
        # plt.show()
        save_fig(
            plt,
            f"../designated-freeriders/"
            f"relative-satisfaction-{ruleclass.replace(" ", "-")}"
            f"-{extra_info_from_filename(filename)}",
        )

        plt.ylabel("satisfaction")
        ax.plot(
            xs,
            [results[rule_index]["satisfaction"][0] for rule_index in range(len(xs))],
            "r-.",
            label="minority freerider: without free-riding",
        )
        ax.plot(
            xs,
            [
                results[rule_index]["satisfaction"][num_voters - 1]
                for rule_index in range(len(xs))
            ],
            "g:",
            label="majority freerider: without free-riding",
        )
        plt.legend(loc="best")
        save_fig(
            plt,
            f"../designated-freeriders/"
            f"satisfaction-{ruleclass.replace(" ", "-")}-{extra_info_from_filename(filename)}",
        )
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            xs,
            [results[rule_index]["freeriding_pos"][0] for rule_index in range(len(xs))],
            "g--",
            label="minority freerider: successful free-riding",
        )
        ax.plot(
            xs,
            [
                results[rule_index]["freeriding_pos"][num_voters - 1]
                for rule_index in range(len(xs))
            ],
            "b-",
            label="majority freerider: successful free-riding",
        )
        ax.plot(
            xs,
            [results[rule_index]["freeriding_neg"][0] for rule_index in range(len(xs))],
            "r-.",
            label="minority freerider: harmful free-riding",
        )
        ax.plot(
            xs,
            [
                results[rule_index]["freeriding_neg"][num_voters - 1]
                for rule_index in range(len(xs))
            ],
            "c:",
            label="majority freerider: harmful free-riding",
        )
        # plt.xlim(-.02)
        plt.title(f"{ruleclass}, {distribution_name(filename)} distribution")
        plt.ylabel("proportion of instances")
        ax.grid(visible=True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # plt.ylim(-0.02, 0.66)
        plt.xlabel("Parameter x")
        if rule == rules.thiele:
            ticks = list(range(8))
            dic = {0: "0\nutilit.", 1: "1\nPAV"}
        elif rule == rules.owa:
            ticks = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
            dic = {0: "0\nutilit.", 19: f"{19}\nleximin"}
        else:
            raise RuntimeError

        labels = [
            ticks[i] if t not in dic.keys() else dic[t] for i, t in enumerate(ticks)
        ]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

        # ax.axvline(1, ls="--")
        # plt.text(20,0,'PAV',rotation=90)
        plt.legend(loc="upper left")
        # plt.show()
        save_fig(
            plt,
            f"../designated-freeriders/occurence-{ruleclass.replace(" ", "-")}"
            f"-{extra_info_from_filename(filename)}",
        )
        plt.close()

        # Draw the violin plot
        if rule == rules.thiele:
            delta_satisfaction_scores = results[PAV_INDEX]["delta_satisfaction_scores"]
            rule_str = "PAV"

        elif rule == rules.owa:
            delta_satisfaction_scores = results[num_voters - 1][
                "delta_satisfaction_scores"
            ]  # leximin
            rule_str = "leximin"
        else:
            raise RuntimeError

        df = pd.DataFrame(
            {
                "delta_satisfaction": delta_satisfaction_scores[0]
                + delta_satisfaction_scores[num_voters - 1],
                "group": ["minority"] * (len(delta_satisfaction_scores[0]))
                + ["majority"] * (len(delta_satisfaction_scores[num_voters - 1])),
            }
        )
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            x="group",
            y="delta_satisfaction",
            data=df,
            showmeans=True,
            meanline=True,
            meanprops=dict(color="black", linestyle=":", linewidth=2.5),
            medianprops=dict(linestyle="--", color="black", linewidth=2.5),
            boxprops=dict(linewidth=1.5, facecolor="none"),
        )
        plt.title(f"{rule_str}, {distribution_name(filename)} distribution")
        plt.ylabel("change in satisfaction")
        save_fig(
            plt,
            "../designated-freeriders/"
            f"satisfaction-boxplot-{rule_str}-{extra_info_from_filename(filename)}",
        )
        plt.close()
        print(f"{filename} done.")


def relative_satisfaction_comparison():
    delta_satisfaction = {}
    for freeriding_model in ["single", "repeated"]:
        parameter_settings = f"different_rules-{freeriding_model}-20issues-5alternatives-20voters-2000-instances"
        relevant_files = [
            filename
            for filename in os.listdir("../")
            if filename.endswith(".pickle") and filename.startswith(parameter_settings)
        ]
        xs = {}
        for fn in relevant_files:
            data = pickle.load(open(f"../{fn}", "rb"))
            for (
                ruleclass,
                rule,
                _xs,
                results,
                num_instances,
                num_voters,
                plot_type,
            ) in data:
                if f"with-approvalradius-{APPROVAL_RADIUS_DEFAULT}" not in fn:
                    continue
                if ruleclass not in delta_satisfaction:
                    delta_satisfaction[ruleclass] = {}
                if freeriding_model not in delta_satisfaction[ruleclass]:
                    delta_satisfaction[ruleclass][freeriding_model] = {}
                if (
                    distribution_name(fn)
                    in delta_satisfaction[ruleclass][freeriding_model]
                ):
                    raise RuntimeError
                delta_satisfaction[ruleclass][freeriding_model][
                    distribution_name(fn)
                ] = [
                    np.mean(
                        [
                            results[rule_index]["delta_satisfaction"][vi]
                            for vi in range(num_voters)
                        ]
                    )
                    for rule_index in range(len(_xs))
                ]
                if ruleclass in xs:
                    assert xs[ruleclass] == _xs
                else:
                    xs[ruleclass] = _xs
    for ruleclass in delta_satisfaction.keys():
        fig, ax = plt.subplots(figsize=(8, 4))
        for freeriding_model in delta_satisfaction[ruleclass].keys():
            for distribution in delta_satisfaction[ruleclass][freeriding_model].keys():
                color = {
                    "square": "#333333",
                    "many groups": "tab:blue",
                    "unbalanced": "tab:red",
                }
                if freeriding_model == "single":
                    freeriding_model_str = "single-issue"
                    linestyle = ":"
                elif freeriding_model == "repeated":
                    freeriding_model_str = "repeated"
                    linestyle = "-"
                else:
                    raise RuntimeError
                ax.plot(
                    xs[ruleclass],
                    delta_satisfaction[ruleclass][freeriding_model][distribution],
                    label=f"{distribution}, {freeriding_model_str}",
                    linestyle=linestyle,
                    color=color[distribution],
                )
        plt.legend()
        plt.title(f"{ruleclass}")
        plt.ylabel("average change in satisfaction")
        ax.grid(visible=True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.ylim(-0.02, 1.64)
        plt.xlabel("Parameter x")
        if ruleclass == "Sequential Thiele methods":
            ticks = list(range(8))
            dic = {0: "0\nutilit.", 1: "1\nPAV"}
        elif ruleclass == "Sequential OWA methods":
            ticks = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
            dic = {0: "0\nutilit.", 19: f"{19}\nleximin"}
        else:
            raise RuntimeError

        labels = [
            ticks[i] if t not in dic.keys() else dic[t] for i, t in enumerate(ticks)
        ]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

        save_fig(
            plt,
            f"../delta_satisfaction/"
            f"satisfaction-comparison-{ruleclass.replace(" ", "-")}"
            f"-{parameter_settings.removeprefix("different_rules-")}",
        )

    print("delta satisfaction plot done.")


def plot_different_num_variable(data):
    if "different_num_voters" in filename and len(data) > 2:
        data = data[-2:]  # quick fix for an earlier bug
    for ruleclass, rule, xs, results, num_instances, num_voters, plot_type in data:
        if num_voters is None:
            num_voter_vals = xs
        else:
            num_voter_vals = [num_voters] * len(xs)

        fig, ax = plt.subplots(figsize=(8, 4))
        assert len(results) == len(xs)
        assert all(len(results[num_var]) == 1 for num_var in range(len(xs)))
        freeriding_pos = [
            np.mean(
                [
                    results[num_var][0]["freeriding_pos"][vi]
                    for vi in range(num_voter_vals[num_var])
                ]
            )
            for num_var in range(len(xs))
        ]
        ax.plot(
            xs,
            freeriding_pos,
            "g--",
            label="likelihood of successful free-riding",
        )
        freeriding_neg = [
            np.mean(
                [
                    results[num_var][0]["freeriding_neg"][vi]
                    for vi in range(num_voter_vals[num_var])
                ]
            )
            for num_var in range(len(xs))
        ]
        ax.plot(
            xs,
            freeriding_neg,
            "r-",
            label="likelihood of harmful free-riding",
        )
        risk = [
            np.mean(
                [
                    results[num_var][0]["risk"][vi]
                    for vi in range(num_voter_vals[num_var])
                ]
            )
            for num_var in range(len(xs))
        ]
        ax.plot(xs, risk, "k-.", label="risk")
        # plt.xlim(-.02)

        if ruleclass == "Sequential Thiele methods":
            plt.title(f"sequ. PAV, {distribution_name(filename)} distribution")
        elif ruleclass == "Sequential OWA methods":
            plt.title(f"sequ. leximin, {distribution_name(filename)} distribution")
        # plt.ylabel("Proportion of voters (average)")
        ax.grid(visible=True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if "-single-" in filename:
            plt.ylim(-0.02, 0.66)
            if plot_type == "different_num_issues":
                plt.ylim(-0.02, 0.96)
        elif "-repeated-" in filename:
            plt.ylim(-0.02, 0.96)
        else:
            raise RuntimeError

        ticks = xs
        if plot_type == "different_num_voters":
            ticks = [x for x in xs if x % 5 == 0]
        dic = {}
        label = plot_type.removeprefix("different_num_")
        if label == "cand":
            label = "candidates"
        plt.xlabel(label)

        labels = [
            ticks[i] if t not in dic.keys() else dic[t] for i, t in enumerate(ticks)
        ]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

        plt.legend(loc="upper left")
        save_fig(
            plt,
            f"../{plot_type}/{ruleclass.replace(" ", "-")}"
            f"-{extra_info_from_filename(filename)}",
        )
        plt.close()
    print(f"{filename} done.")


if __name__ == "__main__":
    written_files = set()

    relative_satisfaction_comparison()
    for filename in os.listdir("../"):
        if not filename.endswith(".pickle"):
            continue
        data = pickle.load(open(f"../{filename}", "rb"))

        if filename.startswith("designated_freerider"):
            plot_designated_freerider(data)
            continue
        elif filename.startswith("different_rules"):
            plot_single_freeriders(data)
            continue
        elif filename.startswith("different_num_"):
            plot_different_num_variable(data)
        else:
            raise RuntimeError

    print(f"generated {len(written_files)} plots")
