# a simple example to check correctness

from profile import MultiIssueProfile
from rules import thiele, owa
import freeriding


def print_freeriding_results(results):
    print("results of freeriding of voter 0 in each issue:")
    for issue_index, result in enumerate(results):
        if result == 1:
            print(f"- issue {issue_index}: freeriding possible and successful")
        elif result == -1:
            print(f"- issue {issue_index}: freeriding possible but harmful")
        elif result == 0:
            print(f"- issue {issue_index}: freeriding possible but with effect")
        elif result is None:
            print(f"- issue {issue_index}: freeriding not possible")
        else:
            raise RuntimeError
    print()


# first example with sequential PAV

miprof = MultiIssueProfile(num_voters=4)
miprof.add_issue([[0], [0], [0], [1]])
miprof.add_issue([[1], [1], [2], [0]])
miprof.add_issue([[0], [1], [0], [1]])
miprof.add_issue([[0], [0], [1], [1]])

pav_weights = [1 / (i + 1) for i in range(4)]
outcome = thiele(miprof, pav_weights)
print(f"outcome: {outcome}")

result_dict = freeriding.freeriding_possible_for_specific_voter(
    miprof, thiele, pav_weights, voter_index=0, freerider_model="single"
)
print_freeriding_results(result_dict["freeriding_per_issue"])


# second example: with sequential egalitarian

miprof = MultiIssueProfile(num_voters=5)
miprof.add_issue([[1], [1], [1], [0], [0]])
miprof.add_issue([[0], [0], [0], [0], [0]])
miprof.add_issue([[1], [0], [0], [1], [1]])
miprof.add_issue([[1], [1], [2], [0], [0]])
miprof.add_issue([[1], [0], [1], [1], [1]])

egal_weights = [1, 0, 0, 0, 0]
outcome = owa(miprof, egal_weights)
print(f"outcome: {outcome}")

result_dict = freeriding.freeriding_possible_for_specific_voter(
    miprof, owa, egal_weights, voter_index=0, freerider_model="single"
)
print_freeriding_results(result_dict["freeriding_per_issue"])
