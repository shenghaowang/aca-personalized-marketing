import matplotlib.pyplot as plt
import numpy as np


def _uplift(responses_control, responses_target, n_control, n_target):
    if n_control == 0:
        return responses_target
    else:
        return responses_target - responses_control * n_target / n_control


def uplift_curve(y_true, d_pred, group, n_nodes=None):
    if n_nodes is None:
        n_nodes = min(len(y_true) + 1, 201)

    sorted_ds = sorted(zip(d_pred, group, y_true), reverse=True)
    responses_control, responses_target, n_control, n_target = 0, 0, 0, 0
    cumulative_responses = [(responses_control, responses_target, n_control, n_target)]

    for _, is_target, response in sorted_ds:
        if is_target:
            n_target += 1
            responses_target += response
        else:
            n_control += 1
            responses_control += response
        cumulative_responses.append(
            (responses_control, responses_target, n_control, n_target)
        )

    xs = [int(i) for i in np.linspace(0, len(y_true), n_nodes)]
    ys = [_uplift(*cumulative_responses[x]) for x in xs]

    return xs, ys


def number_responses(y_true, group):

    responses_target, responses_control, n_target, n_control = 0, 0, 0, 0
    for is_target, y in zip(group, y_true):
        if is_target:
            n_target += 1
            responses_target += y
        else:
            n_control += 1
            responses_control += y

    rescaled_responses_control = (
        0 if n_control == 0 else responses_control * n_target / n_control
    )

    return responses_target, rescaled_responses_control


def plot_uplift_curve(y_true, d_pred, group, n_nodes=None):
    _, ax = plt.subplots(figsize=(10, 7))

    xs, ys = uplift_curve(y_true, d_pred, group, n_nodes=n_nodes)
    ax.plot(xs, ys, label="Model", color="blue")

    # random model
    responses_target, rescaled_responses_control = number_responses(y_true, group)
    incr_responses = responses_target - rescaled_responses_control
    ax.plot(
        [0, len(y_true)],
        [0, incr_responses],
        label="Random",
        color="green",
        linestyle="--",
    )

    # # perfect model
    # sorted_responses = sorted(y_true, reverse=True)
    # cum_responses = np.cumsum(sorted_responses)
    # perfect_ys = [cum_responses[i] if i < n_treated else cum_responses[n_treated - 1] for i in range(len(y_true))]
    # ax.plot(range(len(y_true)), perfect_ys, label="Perfect", color="red", linestyle="--")

    ax.set_xlabel("Number of individuals targeted")
    ax.set_ylabel("Cumulative uplift")
    ax.legend()
    ax.grid(True)

    # Export the figure
    plt.tight_layout()
    plt.savefig("uplift_curve.png")

    return ax
