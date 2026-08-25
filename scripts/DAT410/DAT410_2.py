"""DAT410_2.py — figures for Part 2, Recommender Systems."""

import numpy as np

from rdf import figure

# Purchase probability per user and book.
# Each user has one clear favorite; book C is the best single book on average.
BOOKS = ["A", "B", "C"]
USERS = ["User 1", "User 2", "User 3"]
PROBS = np.array(
    [
        [0.60, 0.20, 0.35],  # User 1 favors A
        [0.10, 0.65, 0.40],  # User 2 favors B
        [0.15, 0.25, 0.55],  # User 3 favors C
    ]
)


@figure("average_vs_personalized")
def plot_average_vs_personalized(ax, color_map):
    """
    Purchase probabilities per user and book.

    Each user's favorite bar is highlighted; the dashed lines compare the
    average sales of the best single book (uniform strategy) against the
    average sales when each user is shown their own favorite (personalized).
    """
    n_users, n_books = PROBS.shape
    width = 0.24

    favorites = PROBS.argmax(axis=1)
    best_single = PROBS.mean(axis=0).max()
    personalized = PROBS.max(axis=1).mean()

    for i in range(n_users):
        for j in range(n_books):
            color = color_map["c1"] if j == favorites[i] else color_map["c7"]
            ax.bar(i + (j - 1) * width, PROBS[i, j], width * 0.9, color=color)
            ax.text(
                i + (j - 1) * width,
                -0.04,
                BOOKS[j],
                ha="center",
                va="top",
                fontsize=8,
            )

    ax.axhline(
        personalized,
        color=color_map["c1"],
        linestyle="--",
        linewidth=1.5,
        label="Average sales, individual favorite",
    )
    ax.axhline(
        best_single,
        color=color_map["c7"],
        linestyle="--",
        linewidth=1.5,
        label="Average sales, best single book",
    )

    ax.set_xticks(range(n_users))
    ax.set_xticklabels(USERS)
    ax.tick_params(axis="x", pad=18, length=0)
    ax.set_ylabel("Purchase probability")
    ax.set_ylim(0, 0.85)
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=8)


@figure("purchase_logs")
def plot_purchase_logs(ax, color_map):
    """
    Simplified purchase logs: which of the books A, B, C each user bought.
    """
    purchases = np.array(
        [
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]
    )
    n_users, n_books = purchases.shape

    for i in range(n_users + 1):
        ax.axhline(i, color=color_map["c7"], linewidth=0.8)
    for j in range(n_books + 1):
        ax.axvline(j, color=color_map["c7"], linewidth=0.8)

    for i in range(n_users):
        for j in range(n_books):
            if purchases[i, j]:
                ax.scatter(
                    j + 0.5,
                    n_users - i - 0.5,
                    marker="X",
                    s=180,
                    color=color_map["c1"],
                )

    ax.set_xticks([j + 0.5 for j in range(n_books)])
    ax.set_xticklabels([f"Book {b}" for b in BOOKS])
    ax.set_yticks([n_users - i - 0.5 for i in range(n_users)])
    ax.set_yticklabels([f"User {i + 1}" for i in range(n_users)])
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.set_xlim(-0.02, n_books + 0.02)
    ax.set_ylim(-0.02, n_users + 0.02)
    ax.set_aspect("equal")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
